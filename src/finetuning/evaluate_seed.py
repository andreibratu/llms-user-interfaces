import json
import os
from pathlib import Path

import tqdm
from joblib import hash as joblib_hash

import src.session as SESSION
from src.configuration import APP_CONFIG
from src.dataset import read_seed_dataset
from src.llm.openai import OpenAILLM
from src.plan.executor import solve_query
from src.plan.planner import SeedPlanner


def evaluate_seed_datasets():
    """Execute human annotated seed datasets and use the output to finetune models.

    The function saves the query evaluations, such as car states and
    memory information, to a file.
    """
    LLM = OpenAILLM(
        APP_CONFIG.experiment.openai_model,
        finetune_strategy="none",
        finetune_format=None,
    )
    SESSION.LLM = LLM

    for plan_format in APP_CONFIG.experiment.plan_formats:
        seed_planner = SeedPlanner(LLM, plan_format)
        dataset = read_seed_dataset(plan_format)

        num_plans = 0
        for query, seed_plans in dataset.items():
            num_plans += len(seed_plans)

        evaluation_file = Path("data", "seed", "evaluation", f"{plan_format}.json")
        evaluations: dict[str, dict] = {}
        os.makedirs(evaluation_file.parent, exist_ok=True)
        if evaluation_file.exists():
            with open(evaluation_file, "r", encoding="utf-8") as fp:
                evaluations = json.load(fp)
                if len(evaluations) == num_plans:
                    print(
                        'Seed dataset for "{}" already evaluated.'.format(plan_format)
                    )
                    continue

        with tqdm.tqdm(total=num_plans, desc=plan_format) as progress_bar:
            for query, seed_plans in dataset.items():
                for plan in seed_plans:
                    query_plan_id = joblib_hash(
                        f"{len(query)}{query}|{len(plan)}{plan}"
                    )
                    assert query_plan_id is not None

                    if query_plan_id in evaluations:
                        progress_bar.update(1)
                        continue

                    seed_planner.inject_query_plan(query, plan)
                    solve_query(query, seed_planner)
                    if seed_planner.evaluation and seed_planner.evaluation.success:
                        evaluations[query_plan_id] = (
                            seed_planner.evaluation.model_dump()
                        )

                    with open(evaluation_file, "w+", encoding="utf-8") as fp:
                        json.dump(evaluations, fp, ensure_ascii=False)

                    progress_bar.update(1)
