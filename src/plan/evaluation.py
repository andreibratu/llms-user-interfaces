import difflib
import json
import string
from collections import Counter
from functools import cached_property
from math import floor
from statistics import mean
from typing import Any, ClassVar, Literal, Optional

import nltk
from Levenshtein import ratio as levenshtein_ratio
from nltk.stem import PorterStemmer
from pydantic import BaseModel, ConfigDict, computed_field

from src.car_state import CarState
from src.configuration import APP_CONFIG
from src.plan.domain import PlanStep
from src.plan.exceptions import BenchmarkException


def _summarize_car_state_for_alignment(
    car_state: dict[str, Any],
) -> dict[str, Any]:
    def _simplify_string(text: str) -> str:
        stopwords = set(nltk.corpus.stopwords.words("english"))
        text = text.lower()
        stemmer = PorterStemmer()
        text = "".join([char for char in text if char not in string.punctuation])
        text = " ".join([word for word in text.split() if word not in stopwords])
        text = " ".join([stemmer.stem(word) for word in text.split()])
        return text.lower()

    result = {}
    for attribute, value in car_state.items():
        if attribute in APP_CONFIG.experiment.alignment_skip_list:
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
    error: BenchmarkException | None = None
    predicted_end_state_alignment: Optional[CarState] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    def mean_query_alignment_score(
        self, mode: Literal["successful", "failed", "all"]
    ) -> float:
        """Compute alignment metric for all successful, failed or either attempts found in a QueryEvaluation."""
        query_alignment_scores = []
        for attempt in self.attempts:
            if attempt.predicted_end_state_alignment is None:
                # Failed to predict end state
                continue

            if mode == "successful" and not attempt.successful:
                # Skip failed attempts
                continue

            if mode == "failed" and attempt.successful:
                # Skip successful attempts
                continue

            actual_end_state = attempt.car_states[-1].model_dump()
            predicted_end_state = attempt.predicted_end_state_alignment.model_dump()

            actual_end_state = _summarize_car_state_for_alignment(actual_end_state)
            predicted_end_state = _summarize_car_state_for_alignment(
                predicted_end_state
            )

            actual_end_state = json.dumps(
                actual_end_state, sort_keys=True, ensure_ascii=False
            )
            predicted_end_state = json.dumps(
                predicted_end_state, sort_keys=True, ensure_ascii=False
            )

            query_alignment_scores.append(
                levenshtein_ratio(actual_end_state, predicted_end_state)
            )
        if len(query_alignment_scores) == 0:
            return 0
        return mean(query_alignment_scores)


class LLMPlannerResult(BaseModel):
    identifier: str
    num_demonstrations: int
    error_feedback_strategy: str
    is_online_strategy: bool
    plan_format: str
    finetuning_strategy: str
    generation_strategy_name: str
    retry_strategy_name: str
    retry_times: int
    query_evaluations: list[QueryEvaluation]

    METRICS: ClassVar[list[str]] = [
        "success_rate",
        "mean_tokens",
        "mean_execution_time",
        "mean_alignment_dist",
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
    def count_errors(self) -> dict[str, int]:
        attempts_errors = [
            attempt.error
            for query in self.query_evaluations
            for attempt in query.attempts
            if attempt.error
        ]
        return dict(Counter([err.code.name for err in attempts_errors]))

    @property
    def mean_plan_size(self) -> float | None:
        lengths = [
            len(att.executed_plan)
            for query in self.query_evaluations
            for att in query.attempts
        ]
        if len(lengths) == 0:
            return None
        return mean(lengths)

    @property
    def mean_tokens(self) -> float:
        tokens_per_attempt = [
            att.used_tokens for eval in self.query_evaluations for att in eval.attempts
        ]
        if len(tokens_per_attempt) == 0:
            return 0
        return float(mean(tokens_per_attempt))

    @property
    def mean_execution_time(self) -> int:
        time_per_attempt = [
            att.time_taken_ms
            for eval in self.query_evaluations
            for att in eval.attempts
        ]
        if len(time_per_attempt) == 0:
            return 0
        return floor(mean(time_per_attempt))

    def compute_metrics(self) -> dict[str, float | int | None]:
        all_metrics = {}
        for metric in self.METRICS:
            output_metric = getattr(self, metric)
            if isinstance(output_metric, dict):
                all_metrics = {**all_metrics, **output_metric}
            else:
                all_metrics[metric] = output_metric
        return all_metrics

    @property
    def mean_alignment_dist(self) -> dict[str, float]:
        """
        Compute alignment metric for all successful, failed or either attempts found in a PlannerResult.
        """
        alignment_scores_all = []
        alignment_scores_success = []
        alignment_scores_failed = []
        for q_eval in self.query_evaluations:
            if q_eval.mean_query_alignment_score is not None:
                alignment_scores_all.append(
                    q_eval.mean_query_alignment_score(mode="all")
                )
                alignment_scores_failed.append(
                    q_eval.mean_query_alignment_score(mode="failed")
                )
                alignment_scores_success.append(
                    q_eval.mean_query_alignment_score(mode="successful")
                )
        return {
            "mean_alignment_all": mean(alignment_scores_all),
            "mean_alignment_success": mean(alignment_scores_success),
            "mean_alignment_failed": mean(alignment_scores_failed),
        }
