import difflib
import json
import string
from collections import Counter
from functools import cached_property
from math import floor
from statistics import mean
from typing import Any, ClassVar, dict, list, Optional, tuple, Union

import nltk
from Levenshtein import ratio as levenshtein_ratio
from nltk.stem import PorterStemmer
from pydantic import BaseModel, computed_field

from src.car_state import CarState
from src.domain import PlanFormat
from src.plan.domain import PlanStep

import src.session as SESSION

nltk.download("stopwords")


def _summarize_car_state_for_alignment(
    car_state: dict[str, Any],
) -> dict[str, Any]:
    def _simplify_string(text: str) -> list[str]:
        stopwords = set(nltk.corpus.stopwords.words("english"))
        text = text.lower()
        stemmer = PorterStemmer()
        text = "".join([char for char in text if char not in string.punctuation])
        text = " ".join([word for word in text.split() if word not in stopwords])
        text = " ".join([stemmer.stem(word) for word in text.split()])
        return text.lower()

    result = {}
    for attribute, value in car_state.items():
        if attribute in SESSION.APP_CONFIG.experiment.alignment_skip_list:
            result[attribute] = value
            continue
        if isinstance(value, dict):
            result[attribute] = _summarize_car_state_for_alignment(value)
            continue
        if isinstance(value, str):
            result[attribute] = _simplify_string(value)
            continue
        if isinstance(value, list):
            new_v = []
            for sub_v in value:
                if isinstance(sub_v, str):
                    sub_v = _simplify_string(sub_v)
                new_v.append(sub_v)
            result[attribute] = new_v
            continue
        result[attribute] = value
    return result


class QueryAttempt(BaseModel):
    raw_llm_text: list[str] = []
    intended_plan: list[PlanStep] = []
    executed_plan: list[PlanStep] = []
    car_states: list[CarState] = []
    tool_output: list[tuple[str, str]] = []
    memory_states: list[dict] = []
    used_tokens: int = 0
    time_taken_ms: int = 0
    error: Optional[dict] = None
    predicted_state_alignment: Optional[CarState] = None

    @computed_field
    @cached_property
    def transient_tool_outputs(self) -> list[tuple[str, str]]:
        """Return tool outputs from tools that depend on time of execution."""
        return [
            tool_output_pair
            for tool_output_pair in self.tool_output
            if tool_output_pair[0] in ["weather_tool", "get_current_date"]
        ]

    @computed_field
    @property
    def successful(self) -> bool:
        return self.error is None


_DIFFER = difflib.Differ()


class QueryEvaluation(BaseModel):
    query: str
    attempts: list[QueryAttempt]

    @computed_field
    def used_tries(self) -> int:
        return len(self.attempts)

    @computed_field
    def success(self) -> bool:
        return any(att.error is None for att in self.attempts)

    @computed_field
    @cached_property
    def alignment_outcome_delta(self) -> Optional[str]:
        """Compute Git-style difference between predicted outcome and actual car state.

        "minus" lines indicate a value present in prediction but not in real outcome
        """
        if (
            self.attempts[-1].car_states[-1] is None
            or self.attempts[-1].predicted_state_alignment is None
        ):
            return None

        actual_end_state = json.dumps(
            self.attempts[-1].car_states[-1].model_dump(),
            sort_keys=True,
            indent=0,
        ).splitlines()
        predicted_end_state = json.dumps(
            self.attempts[-1].predicted_state_alignment.model_dump(),
            sort_keys=True,
            indent=0,
        ).splitlines()

        return "\n".join(
            [
                diff
                for diff in _DIFFER.compare(predicted_end_state, actual_end_state)
                if not diff.startswith(" ")
            ]
        )

    @computed_field
    @cached_property
    def alignment_score(self) -> Optional[float]:
        if (
            self.attempts[-1].car_states[-1] is None
            or self.attempts[-1].predicted_state_alignment is None
        ):
            return None

        actual_end_state = self.attempts[-1].car_states[-1].model_dump()
        predicted_end_state = self.attempts[-1].predicted_state_alignment.model_dump()

        actual_end_state = _summarize_car_state_for_alignment(actual_end_state)
        predicted_end_state = _summarize_car_state_for_alignment(predicted_end_state)

        actual_end_state = json.dumps(actual_end_state, sort_keys=True)
        predicted_end_state = json.dumps(predicted_end_state, sort_keys=True)

        return levenshtein_ratio(actual_end_state, predicted_end_state)


class LLMPlannerResult(BaseModel):
    identifier: str
    plan_format: PlanFormat
    num_demonstrations: int
    use_alignment_prediction: bool
    error_feedback_strategy: str
    is_online_strategy: bool
    strategy_name: str
    llm_name: str
    wire_producers: Optional[str]
    retry_times: int
    query_evaluations: list[QueryEvaluation]

    METRICS: ClassVar[list[str]] = [
        "success_rate",
        "partial_executability_best_effort",
        "partial_executability_mean",
        "mean_tokens",
        "mean_execution_time",
        "mean_success_tokens",
        "mean_success_time",
        "mean_alignment_dist",
        "alignment_success_percentage_viable",
        "alignment_percentage_viable",
        "mean_plan_size",
    ]

    @property
    def success_rate(self) -> float:
        ok, total = 0, 0
        for query in self.query_evaluations:
            if query.success:
                ok += 1
            total += 1
        return ok / total

    @property
    def mean_plan_size(self) -> Optional[float]:
        lengths = [
            len(att.executed_plan)
            for query in self.query_evaluations
            for att in query.attempts
        ]
        if len(lengths) == 0:
            return None
        return mean(lengths)

    @property
    def partial_executability_best_effort(self) -> Union[float, None]:
        """
        Evaluate only the last instance which must be the best attempt of the model.
        """
        if not self.is_online_strategy:
            return None
        exec_best_attempt_query = []
        for query in self.query_evaluations:
            if len(query.attempts) == 0 or len(query.attempts[-1].intended_plan) == 0:
                # JSON could not be parsed for LLM outputted plan
                exec_best_attempt_query.append(0)
            else:
                exec_best_attempt_query.append(
                    len(query.attempts[-1].executed_plan)
                    / len(query.attempts[-1].intended_plan)
                )

        return round(mean(exec_best_attempt_query), 2)

    @property
    def partial_executability_mean(self) -> Union[float, None]:
        if not self.online_strategy:
            return None
        avg_exec_per_query = []
        for q_eval in self.query_evaluations:
            attempts = []
            for att in q_eval.attempts:
                if len(att.intended_plan) == 0:
                    # JSON could not be parsed for LLM outputted plan
                    attempts.append(0)
                else:
                    attempts.append(len(att.executed_plan) / len(att.intended_plan))
            if len(attempts) == 0:
                avg_exec_per_query.append(0)
            else:
                avg_exec_per_query.append(mean(attempts))
        return round(mean(avg_exec_per_query), 2)

    @property
    def count_errors(self) -> dict[str, int]:
        attempts_errors: list[Optional[dict]] = [
            att.error
            for query in self.query_evaluations
            for att in query.attempts
            if att.error
        ]
        return Counter([(err["code"], err["taxonomy"]) for err in attempts_errors])

    @property
    def mean_tokens(self) -> float:
        tokens_per_attempt = [
            att.used_tokens for eval in self.query_evaluations for att in eval.attempts
        ]
        if len(tokens_per_attempt) == 0:
            return 0
        return round(mean(tokens_per_attempt), 2)

    @property
    def mean_execution_time(self) -> float:
        time_per_attempt = [
            att.time_taken_ms
            for eval in self.query_evaluations
            for att in eval.attempts
        ]
        if len(time_per_attempt) == 0:
            return 0
        return round(mean(time_per_attempt), 2)

    @property
    def mean_success_tokens(self) -> Optional[float]:
        successful = [
            eval.attempts[-1].used_tokens
            for eval in self.query_evaluations
            if eval.success
        ]
        if len(successful) == 0:
            return None
        return floor(mean(successful))

    @property
    def mean_success_time(self) -> Optional[float]:
        successful = [
            eval.attempts[-1].time_taken_ms
            for eval in self.query_evaluations
            if eval.success
        ]
        if len(successful) == 0:
            return None
        return floor(mean(successful))

    def compute_metrics(self) -> dict[str, Union[float, int, None]]:
        all_metrics = {}
        for metric in self.METRICS:
            output_metric = getattr(self, metric)
            if isinstance(output_metric, dict):
                all_metrics = {**all_metrics, **output_metric}
            else:
                all_metrics[metric] = output_metric
        return all_metrics

    @property
    def alignment_percentage_viable(self) -> float:
        """
        Ratio of pairs where the (actual_state, alignment_predicted)
        are both non-null out of all evaluations.
        """
        criteria_count = sum(
            1
            for eval in self.query_evaluations
            if len(eval.attempts[-1].car_states) != 0
            and eval.attempts[-1].predicted_state_alignment
        )
        return criteria_count / len(self.query_evaluations)

    @property
    def alignment_success_percentage_viable(self) -> float:
        """
        Ratio of pairs where the (actual_state, alignment_predicted)
        are both non-null in successful evaluations out of all evaluations.

        Should mismatch with the success rate of the alignment state
        prediction function returns null.
        """
        criteria_count = sum(
            1
            for eval in self.query_evaluations
            if eval.success and eval.attempts[-1].predicted_state_alignment
        )
        return criteria_count / len(self.query_evaluations)

    @property
    def mean_alignment_dist(self) -> dict[str, Optional[float]]:
        """Compute alignment over all queries. For failed queries, will use the state
        of the last attempt (assumed to be best since feedback was received on
        past mistakes), and the latest car state of that query.
        """
        alignment_scores = []
        alignment_scores_success = []
        for q_eval in self.query_evaluations:
            if q_eval.alignment_score is not None:
                if q_eval.success:
                    alignment_scores_success.append(q_eval.alignment_score)
                alignment_scores.append(q_eval.alignment_score)
        if len(alignment_scores) == 0:
            return {"mean_alignment_all": None, "mean_alignment_success": None}
        return {
            "mean_alignment_all": mean(alignment_scores),
            "mean_alignment_success": mean(
                [score for score in alignment_scores if score is not None]
            ),
        }
