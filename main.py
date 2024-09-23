import itertools
import json
import random
from typing import Iterable, list, tuple

import numpy as np
import tqdm

import src.session as SESSION
from src.database import planner_evaluated, write_metrics
from src.dataset.benchmark import generate_benchmark_dataset
from src.llm import LLMInterface
from src.llm.openai import OpenAILLM
from src.plan.evaluation import LLMPlannerResult
from src.plan.executor import solve_query
from src.plan.planner import LLMPlanner, QueryEvaluation
from src.prompts import DEFAULT_LLM_HYPERS
from src.strategy.generate.generate_blind import GenerateBlindOffline
from src.strategy.generate.generate_strategy import GenerateStrategy
from src.strategy.retry import RetryStrategy, TryManyTimes

random.seed(SESSION.APP_CONFIG.experiment.random_seed)


generate_benchmark_dataset(
    generator_llm=SESSION.LLM,
    num_instructions_generate=SESSION.APP_CONFIG.generation.generate_size,
)

with open("data/dataset.jsonl", "r", encoding="utf-8") as fp:
    lines = fp.readlines()

_LLMS = [
    OpenAILLM(model=model) for model in SESSION.APP_CONFIG.experiment.openai_models
]
_EVAL_DATASET = random.sample(lines, SESSION.APP_CONFIG.experiment.dataset_size)
_RETRY_STRATEGIES: list[RetryStrategy] = [
    TryManyTimes(retry_time) for retry_time in [1, 3]
]
_GENERATE_STRATEGIES: list[GenerateStrategy] = []


# for (
#     llm,
#     num_demonstrations,
#     feedback_strategy,
#     use_align_pred,
#     plan_output_mode,
# ) in itertools.product(
#     _LLMS,
#     APP_CONFIG.experiment.num_demonstrations,
#     APP_CONFIG.experiment.feedback_strategies,
#     APP_CONFIG.experiment.use_alignment_prediction,
#     APP_CONFIG.experiment.plan_output_mode,
# ):
#     init_args = {
#         "planner_llm": llm,
#         "llm_hypers": DEFAULT_LLM_HYPERS,
#         "err_feedback_strategy": feedback_strategy,
#         "num_demonstrations": num_demonstrations,
#         "use_alignment_prediction": use_align_pred,
#         "plan_output_mode": plan_output_mode,
#     }
#     if "gml" in plan_output_mode:
#         _GENERATE_STRATEGIES.extend(
#             [
#                 GenerateBlindOffline(**init_args),
#                 GenerateCOT(**init_args),
#             ]
#         )
#     else:
#         _GENERATE_STRATEGIES.extend(
#             [
#                 GenerateBlindOnline(**init_args),
#                 GenerateBlindOffline(**init_args),
#                 GenerateReact(**init_args),
#                 GenerateCOT(**init_args),
#                 GenerateBabyAGI(**init_args),
#             ]
#         )

_GENERATE_STRATEGIES = [
    GenerateBlindOffline(
        planner_llm=_LLMS[0],
        llm_hypers=DEFAULT_LLM_HYPERS,
        num_demonstrations=10,
        err_feedback_strategy="ERROR_TYPE+STEP",
        use_alignment_prediction=True,
        is_online_strategy=False,
        plan_format="json",
        strategy_name="GenerateBlindOffline",
    ),
    GenerateBlindOffline(
        planner_llm=_LLMS[0],
        llm_hypers=DEFAULT_LLM_HYPERS,
        num_demonstrations=10,
        err_feedback_strategy="ERROR_TYPE+STEP",
        use_alignment_prediction=True,
        is_online_strategy=False,
        plan_format="gml",
        strategy_name="GenerateBlindOffline",
    ),
]

experiment_it: Iterable[tuple[LLMInterface, RetryStrategy, GenerateStrategy]] = (
    itertools.product(
        _LLMS,
        _RETRY_STRATEGIES,
        _GENERATE_STRATEGIES,
    )
)
num_configurations = np.prod(
    [len(_LLMS), len(_RETRY_STRATEGIES), len(_GENERATE_STRATEGIES)]
)
for llm, rs, gs in (
    epb := tqdm.tqdm(
        iterable=experiment_it,
        total=num_configurations,
    )
):
    SESSION.LLM = llm
    planner = LLMPlanner(
        retry_strategy=rs,
        generation_strategy=gs,
        llm=llm,
    )

    for repeat_idx in range(SESSION.APP_CONFIG.experiment.repeat_experiments):
        query_evaluations: list[QueryEvaluation] = []
        trial_id = f"{planner.identifier}_{repeat_idx}"
        if planner_evaluated(trial_id):
            epb.write(f"Planner {trial_id} already evaluated")
            epb.update(1)
            continue

        for query_line in (pb := tqdm.tqdm(_EVAL_DATASET, desc="Dataset")):
            query_text = json.loads(query_line)["instruction"]
            solve_query(query_text, planner)
            query_evaluations.append(planner.evaluation.model_copy())

        write_metrics(
            LLMPlannerResult(
                identifier=trial_id,
                plan_format=gs.plan_format,
                num_demonstrations=gs.num_demonstrations,
                use_alignment_prediction=gs.use_alignment_prediction,
                error_feedback_strategy=gs.error_feedback_strategy,
                is_online_strategy=gs.is_online_strategy,
                strategy_name=gs.strategy_name,
                llm_name=gs.planner_llm.name,
                # Will only return for graph strategies, else None
                wire_producers=gs.__dict__.get("wire_producers"),
                retry_times=rs.__dict__.get("times"),
                query_evaluations=query_evaluations,
            )
        )
