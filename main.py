import itertools
import json
import random
from pathlib import Path
from typing import Iterable, Literal

import tqdm

import src.session as SESSION
from src.car_state import CarState
from src.configuration import APP_CONFIG
from src.dataset.benchmark import generate_benchmark_dataset
from src.domain import FinetunedLLMConfig, PlanFormat
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
from src.strategy.retry import RetryStrategy, TryManyTimes

SESSION.CAR_STATE = CarState.get_default()


random.seed(APP_CONFIG.experiment.random_seed)


LLM_CONFIGS: list[FinetunedLLMConfig] = []
# Get fine-tuned model names
with open(
    Path("data", "finetuning", "finetune_jobs.json"), "r", encoding="utf-8"
) as fp:
    _FINETUNE_JOBS = json.load(fp)
    for key, job_config in _FINETUNE_JOBS.items():
        finetune_strategy, plan_format = key.split("_")
        LLM_CONFIGS.append(
            FinetunedLLMConfig(
                finetune_strategy=finetune_strategy,
                plan_format=plan_format,
                model=job_config["fine_tuned_model"],
            )
        )


LLM_CONFIGS.append(
    FinetunedLLMConfig(
        finetune_strategy="none",
        plan_format=None,
        model=APP_CONFIG.experiment.openai_model,
    )
)

# Use baseline model to generate benchmark dataset
SESSION.LLM = OpenAILLM(
    model=APP_CONFIG.experiment.openai_model,
    finetuning_strategy="none",
)
generate_benchmark_dataset(
    generator_llm=SESSION.LLM,
    num_instructions_generate=APP_CONFIG.generation.generate_size,
)
with open(Path("data", "benchmark", "benchmark.jsonl"), "r", encoding="utf-8") as fp:
    _BENCHMARK_DATASET = random.sample(
        fp.readlines(), APP_CONFIG.experiment.dataset_size
    )


LLMS: list[LLMInterface] = []
RETRY_STRATEGIES: list[TryManyTimes] = [
    TryManyTimes(retry_time) for retry_time in APP_CONFIG.experiment.retry_times
]
GENERATE_STRATEGIES: list[GenerateStrategy] = []


for (
    llm_config,
    num_demonstrations,
    feedback_strategy,
    plan_format,
) in itertools.product(
    LLM_CONFIGS,
    APP_CONFIG.experiment.num_demonstrations,
    APP_CONFIG.experiment.feedback_strategies,
    APP_CONFIG.experiment.plan_formats,
):
    # if llm_finetune_plan_format is None, then it can be used with any plan format
    finetune_strategy, llm_finetune_plan_format, model_name = llm_config
    if llm_finetune_plan_format is not None and llm_finetune_plan_format != plan_format:
        # Skip invalid configuration: LLM finetuned with different plan format
        continue

    llm = OpenAILLM(
        model=llm_config.model,
        finetuning_strategy=llm_config.finetune_strategy,
    )
    LLMS.append(llm)
    init_args = {
        "planner_llm": llm,
        "llm_hypers": DEFAULT_LLM_HYPERS,
        "err_feedback_strategy": feedback_strategy,
        "num_demonstrations": num_demonstrations,
        "plan_format": plan_format,
    }
    if "gml" in plan_format:
        GENERATE_STRATEGIES.extend(
            [
                GenerateBlindOffline(**init_args),
                GenerateCOTOffline(**init_args),
            ]
        )
    else:
        GENERATE_STRATEGIES.extend(
            [
                GenerateBlindOffline(**init_args),
                GenerateBlindOnline(**init_args),
                GenerateCOTOffline(**init_args),
                GenerateCOTOnline(**init_args),
                GenerateReactOnline(**init_args),
            ]
        )


experiment_it: Iterable[tuple[LLMInterface, RetryStrategy, GenerateStrategy]] = (
    itertools.product(
        LLMS,
        RETRY_STRATEGIES,
        GENERATE_STRATEGIES,
    )
)
configurations = list(experiment_it)
for llm, rs, gs in (
    epb := tqdm.tqdm(
        iterable=configurations,
        total=len(configurations),
    )
):
    SESSION.LLM = llm
    planner = LLMPlanner(
        retry_strategy=rs,
        generation_strategy=gs,
        llm=llm,
    )
    if gs.strategy_name != "cot":
        continue
    for repeat_idx in range(APP_CONFIG.experiment.repeat_experiments):
        query_evaluations: list[QueryEvaluation] = []
        trial_id = f"{planner.identifier}_{repeat_idx}"
        if planner_evaluated(trial_id):
            epb.write(f"Planner {trial_id} already evaluated")
            epb.update(1)
            continue

        for query_line in (pb := tqdm.tqdm(_BENCHMARK_DATASET, desc="Dataset")):
            query_text = json.loads(query_line)["instruction"]
            solve_query(query_text, planner)
            query_evaluations.append(planner.evaluation.model_copy())

        write_metrics(
            LLMPlannerResult(
                identifier=trial_id,
                plan_format=gs.plan_format,
                num_demonstrations=gs.num_demonstrations,
                error_feedback_strategy=gs.error_feedback_strategy,
                finetuning_strategy=llm.finetuning_strategy,
                is_online_strategy=gs.is_online_strategy,
                generation_strategy_name=gs.strategy_name,
                retry_strategy_name=rs.strategy_name,
                retry_times=rs.retry_times,
                query_evaluations=query_evaluations,
            )
        )

        write_metrics_report()
