import json
from pathlib import Path

import pandas as pd
from pydantic import TypeAdapter

from src.car_state import CarState
from src.plan.domain import PlanStep
from src.plan.evaluation import LLMPlannerResult


def _insert_attempts(planner_eval: LLMPlannerResult):
    attempts_file = Path("data", "benchmark", "attempts.json")
    if not attempts_file.exists():
        attempts_file.touch()
        attempts = []
    else:
        with open(attempts_file, "r") as fp:
            attempts = json.load(fp)
    attempts.extend(
        [
            {
                "identifier": planner_eval.identifier,
                "plan_format": planner_eval.plan_format,
                "num_demonstrations": planner_eval.num_demonstrations,
                "error_feedback_strategy": planner_eval.error_feedback_strategy,
                "finetuning_strategy": planner_eval.finetune_strategy,
                "is_online_strategy": planner_eval.is_online_strategy,
                "generation_strategy_name": planner_eval.generation_strategy_name,
                "retry_strategy_name": planner_eval.retry_strategy_name,
                "retry_times": planner_eval.retry_times,
                "successful": attempt.successful,
                "query": query_eval.query,
                "car_states": TypeAdapter(list[CarState]).dump_python(
                    attempt.car_states
                ),
                "tool_output": attempt.tool_output,
                "intended_plan": TypeAdapter(list[PlanStep]).dump_python(
                    attempt.intended_plan
                ),
                "executed_plan": TypeAdapter(list[PlanStep]).dump_python(
                    attempt.executed_plan
                ),
                "raw_llm_text": attempt.raw_llm_text,
                "predicted_state_alignment": TypeAdapter(CarState).dump_python(
                    attempt.predicted_end_state_alignment
                )
                if attempt.predicted_end_state_alignment
                else None,
                "memory_states": attempt.memory_states,
            }
            for query_eval in planner_eval.query_evaluations
            for attempt in query_eval.attempts
        ]
    )
    with open(attempts_file, "w") as fp:
        json.dump(attempts, fp, indent=2)


def _insert_errors(planner_eval: LLMPlannerResult):
    errors_file = Path("data", "benchmark", "errors.json")
    if not errors_file.exists():
        errors_file.touch()
        errors = []
    else:
        with open(errors_file, "r") as fp:
            errors = json.load(fp)
    errors.extend(
        [
            {
                "identifier": planner_eval.identifier,
                "plan_format": planner_eval.plan_format,
                "num_demonstrations": planner_eval.num_demonstrations,
                "error_feedback_strategy": planner_eval.error_feedback_strategy,
                "finetuning_strategy": planner_eval.finetune_strategy,
                "is_online_strategy": planner_eval.is_online_strategy,
                "generation_strategy_name": planner_eval.generation_strategy_name,
                "retry_strategy_name": planner_eval.retry_strategy_name,
                "retry_times": planner_eval.retry_times,
                "successful": attempt.successful,
                "query": query_eval.query,
                **{"code": attempt.error.code, "message": attempt.error.message},
            }
            for query_eval in planner_eval.query_evaluations
            for attempt in query_eval.attempts
            if attempt.error
        ]
    )
    with open(errors_file, "w") as fp:
        json.dump(errors, fp, indent=2)


def _write_metrics(planner_eval: LLMPlannerResult):
    metrics_file = Path("data", "benchmark", "metrics.json")
    if not metrics_file.exists():
        metrics_file.touch()
        metrics = []
    else:
        with open(metrics_file, "r") as fp:
            metrics = json.load(fp)
    metrics.append(
        {
            "identifier": planner_eval.identifier,
            "plan_format": planner_eval.plan_format,
            "num_demonstrations": planner_eval.num_demonstrations,
            "error_feedback_strategy": planner_eval.error_feedback_strategy,
            "finetuning_strategy": planner_eval.finetune_strategy,
            "is_online_strategy": planner_eval.is_online_strategy,
            "generation_strategy_name": planner_eval.generation_strategy_name,
            "retry_strategy_name": planner_eval.retry_strategy_name,
            "retry_times": planner_eval.retry_times,
            **planner_eval.compute_metrics(),
        }
    )
    with open(metrics_file, "w") as fp:
        json.dump(metrics, fp, indent=2)


def write_metrics_report():
    metrics_file = Path("data", "benchmark", "metrics.json")
    if not metrics_file.exists():
        assert False, "No metrics to report"
    with open(metrics_file, "r") as fp:
        metrics = json.load(fp)

    df = pd.DataFrame.from_records(
        metrics,
        index="identifier",
    )
    df.to_csv(Path("data", "benchmark", "metrics.csv"))


def planner_evaluated(planner_identifier: str) -> bool:
    evaluated_file = Path("data", "benchmark", "evaluated.json")
    if not evaluated_file.exists():
        return False
    with open(evaluated_file, "r") as fp:
        EVALUATED = json.load(fp)
    return any(planner["identifier"] == planner_identifier for planner in EVALUATED)


def write_metrics(planner_evaluation: LLMPlannerResult):
    evaluated_file = Path("data", "benchmark", "evaluated.json")
    if not evaluated_file.exists():
        evaluated_file.touch()
        EVALUATED = []
    else:
        with open(evaluated_file, "r") as fp:
            EVALUATED = json.load(fp)
    EVALUATED.append(
        {"identifier": planner_evaluation.identifier},
    )
    for fn in [_write_metrics, _insert_errors, _insert_attempts]:
        fn(planner_evaluation)
    with open(evaluated_file, "w") as fp:
        json.dump(EVALUATED, fp, indent=2)
