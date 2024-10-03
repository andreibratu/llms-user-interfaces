import itertools
import json
import random
import typing
from pathlib import Path

import tqdm

import src.session as SESSION
from src.car_state import CarState
from src.configuration import APP_CONFIG
from src.dataset.benchmark import generate_benchmark_dataset
from src.domain import PlanFormat
from src.llm import LLMInterface
from src.llm.openai import OpenAILLM
from src.plan.evaluation import LLMPlannerResult
from src.plan.executor import solve_query
from src.plan.planner import LLMPlanner, QueryEvaluation
from src.prompts import DEFAULT_LLM_HYPERS
from src.report import planner_evaluated, write_metrics, write_metrics_report
from src.strategy.generate import GenerateStrategy
from src.strategy.generate.generate_blind import (
    GenerateBlindOffline,
    GenerateBlindOnline,
)
from src.strategy.generate.generate_cot import GenerateCOTOffline, GenerateCOTOnline
from src.strategy.generate.generate_react import GenerateReactOnline
from src.strategy.retry import TryManyTimes

SESSION.CAR_STATE = CarState.get_default()


random.seed(APP_CONFIG.experiment.random_seed)


LLMS: list[LLMInterface] = [
    # Baseline, no fine-tuning
    OpenAILLM(
        finetune_strategy="none",
        finetune_format=None,
        model=APP_CONFIG.experiment.openai_model,
    )
]
# Get fine-tuned model names
with open(
    Path("data", "finetuning", "finetune_jobs.json"), "r", encoding="utf-8"
) as fp:
    _FINETUNE_JOBS: dict[str, dict] = json.load(fp)
    for key, job_config in _FINETUNE_JOBS.items():
        finetune_strategy, plan_format = key.rsplit("_", 1)
        assert finetune_strategy in typing.get_args(
            typing.Literal["none", "tool_bert", "baseline"]
        )
        assert plan_format in typing.get_args(PlanFormat)
        LLMS.append(
            OpenAILLM(
                finetune_strategy=finetune_strategy,  # pyright: ignore [reportArgumentType]
                finetune_format=plan_format,  # pyright: ignore [reportArgumentType]
                model=job_config["fine_tuned_model"],
            )
        )

# Use baseline model to generate benchmark dataset
SESSION.LLM = OpenAILLM(
    model=APP_CONFIG.experiment.openai_model,
    finetune_strategy="none",
    finetune_format=None,
)
generate_benchmark_dataset(
    generator_llm=SESSION.LLM,
    num_instructions_generate=APP_CONFIG.generation.generate_size,
)
with open(Path("data", "benchmark", "benchmark.jsonl"), "r", encoding="utf-8") as fp:
    EVALUATION_QUERIES = fp.readlines()

# Build configurations to be evaluated
GENERATE_STRATEGIES: list[GenerateStrategy] = []
for (
    llm,
    plan_format,
) in itertools.product(
    LLMS,
    APP_CONFIG.experiment.plan_formats,
):
    # if llm_config.plan_format is None, then it can be used with any plan format
    # It is a non-finetuned model
    if llm.finetune_format is not None and llm.finetune_format != plan_format:
        # Skip invalid configuration: LLM finetuned with different plan format
        continue

    # Sample points from num_demonstrations domain so we can avoid exhaustive
    # grid evaluation
    num_demonstrations = [
        random.randint(
            APP_CONFIG.experiment.num_demonstrations[0],
            APP_CONFIG.experiment.num_demonstrations[1],
        )
        for _ in range(APP_CONFIG.experiment.num_demonstrations_picks)
    ]

    for num_d in num_demonstrations:
        strategy_args = {
            "planner_llm": llm,
            "llm_hypers": DEFAULT_LLM_HYPERS,
            "err_feedback_strategy": random.choice(
                APP_CONFIG.experiment.feedback_strategies
            ),
            "num_demonstrations": num_d,
            "plan_format": plan_format,
        }
        if "gml" in plan_format:
            GENERATE_STRATEGIES.extend(
                [
                    GenerateBlindOffline(**strategy_args),
                    GenerateCOTOffline(**strategy_args),
                ]
            )
        else:
            GENERATE_STRATEGIES.extend(
                [
                    GenerateBlindOffline(**strategy_args),
                    GenerateBlindOnline(**strategy_args),
                    GenerateCOTOffline(**strategy_args),
                    GenerateCOTOnline(**strategy_args),
                    GenerateReactOnline(**strategy_args),
                ]
            )


for gs in (epb := tqdm.tqdm(iterable=GENERATE_STRATEGIES[8:])):
    # Select random retry strategy between 1 and 3
    rs = TryManyTimes(
        random.randint(
            APP_CONFIG.experiment.retry_times[0], APP_CONFIG.experiment.retry_times[1]
        )
    )
    # Distinct LLM attached at earlier loop for each generation strategy
    SESSION.LLM = gs.planner_llm
    planner = LLMPlanner(
        retry_strategy=rs,
        generation_strategy=gs,
        llm=gs.planner_llm,
    )
    # Get fresh batches for each configuration - evaluating the grid
    _EVALUATION_BATCHES = [
        random.sample(EVALUATION_QUERIES, APP_CONFIG.experiment.batch_size)
        for _ in range(APP_CONFIG.experiment.batch_picks)
    ]
    for batch_idx, batch in enumerate(_EVALUATION_BATCHES):
        query_evaluations: list[QueryEvaluation] = []
        trial_id = f"{planner.identifier}_{batch_idx}"
        # TODO: planner_evaluated is less useful when doing sampling over the hyperparameter grid
        if planner_evaluated(trial_id):
            epb.write(f"Planner {trial_id} already evaluated")
            epb.update(1)
            continue

        for query_line in (pb := tqdm.tqdm(batch, desc=f"Dataset_{batch_idx}")):
            query_text = json.loads(query_line)["instruction"]
            solve_query(query_text, planner)
            query_evaluations.append(planner.evaluation.model_copy())

        write_metrics(
            LLMPlannerResult(
                identifier=trial_id,
                plan_format=gs.plan_format,
                num_demonstrations=gs.num_demonstrations,
                error_feedback_strategy=gs.error_feedback_strategy,
                finetune_strategy=gs.planner_llm.finetuning_strategy,
                is_online_strategy=gs.is_online_strategy,
                generation_strategy_name=gs.strategy_name,
                retry_strategy_name=rs.strategy_name,
                retry_times=rs.retry_times,
                query_evaluations=query_evaluations,
            )
        )

        write_metrics_report()
