import os
from pathlib import Path

import pandas as pd
import pymongo

from src.configuration import APP_CONFIG
from src.plan.evaluation import LLMPlannerResult

MONGO = pymongo.MongoClient(
    APP_CONFIG.mongo.connect_url,
    connect=True,
)
DB = MONGO.get_database("evaluator")


METRICS = DB.get_collection("metrics")
ERRORS = DB.get_collection("errors")
ATTEMPTS = DB.get_collection("attempts")
EVALUATED = DB.get_collection("evaluated")


def _insert_attempts(planner_eval: LLMPlannerResult, eval_mode: str):
    ATTEMPTS.insert_many(
        [
            {
                "identifier": planner_eval.identifier,
                "evaluation_mode": eval_mode,
                "successful": attempt.successful,
                "online": planner_eval.is_online_strategy,
                "plan_format": planner_eval.plan_format,
                "strategy_name": planner_eval.strategy_name,
                "query": query_eval.query,
                **attempt.model_dump_json(),
            }
            for query_eval in planner_eval.query_evaluations
            for attempt in query_eval.attempts
        ]
    )


def _insert_errors(planner_eval: LLMPlannerResult, eval_mode: str):
    ERRORS.insert_many(
        [
            {
                "identifier": planner_eval.identifier,
                "evaluation_mode": eval_mode,
                "online": planner_eval.is_online_strategy,
                "plan_format": planner_eval.plan_format,
                "strategy_name": planner_eval.strategy_name,
                "query": query_eval.query,
                **attempt.error,
            }
            for query_eval in planner_eval.query_evaluations
            for attempt in query_eval.attempts
            if attempt.error
        ]
    )


def _write_metrics(planner_eval: LLMPlannerResult):
    # Cannot compute relative executability i.e. percentage of intended plan
    # computed when the plan is generated step by step
    METRICS.insert_one(
        {
            "identifier": planner_eval.identifier,
            **planner_eval.compute_metrics(),
        }
    )


def write_metrics_report():
    os.makedirs(
        benchmark_dir := Path("reports", "benchmark"),
        exist_ok=True,
    )
    all_planners_metrics = METRICS.find(
        projection=["identifier", LLMPlannerResult.METRICS],
    )
    df = pd.DataFrame.from_records(
        all_planners_metrics,
        index="identifier",
    )
    with open(
        benchmark_dir.joinpath("report.txt"),
        "w+",
        encoding="utf=8",
    ) as fp:
        fp.write(df.to_string())


def planner_evaluated(
    planner_identifier: str,
) -> bool:
    return EVALUATED.find_one({"identifier": planner_identifier}) is not None


def write_metrics(planner_evaluation: LLMPlannerResult):
    EVALUATED.insert_one(
        {"identifier": planner_evaluation.identifier},
    )
    for fn in [_write_metrics, _insert_errors, _insert_attempts]:
        fn(planner_evaluation)
