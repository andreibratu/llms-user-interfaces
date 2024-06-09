import json
import os
from pathlib import Path
from typing import Dict
from joblib import hash as joblib_hash
import tqdm
from configuration import APP_CONFIG
from dataset.read_seed import read_seed_dataset
from llm.openai import OpenAILLM
from plan.executor import solve_query
from plan.planner import SeedPlanner
from plan.evaluation import QueryEvaluation
import session as SESSION


if __name__ == "__main__":
    LLM = OpenAILLM("gpt-3.5-turbo-0125")
    SESSION.ORACLE = LLM

    for mode in APP_CONFIG.experiment.plan_output_mode:
        seed_planner = SeedPlanner(LLM, mode)
        dataset = read_seed_dataset(mode)

        size = 0
        for _, seed_plans in dataset.items():
            size += len(seed_plans)

        evaluation_file = Path(__file__).parent.joinpath(
            "data", "seed_evaluate", f"{mode}_dataset_evaluation.json"
        )
        evaluation: Dict[str, QueryEvaluation] = {}
        os.makedirs(evaluation_file.parent, exist_ok=True)
        if evaluation_file.exists():
            with open(evaluation_file, "r") as fp:
                evaluation: Dict[str, QueryEvaluation] = json.load(fp)

        with tqdm.tqdm(total=size, desc=mode) as progress_bar:
            for query, seed_plans in dataset.items():

                for plan in seed_plans:
                    id = joblib_hash(f"{len(query)}{query}|{len(plan)}{plan}")

                    if id in evaluation:
                        progress_bar.update(1)
                        continue

                    seed_planner.inject_query_plan(query, plan)
                    solve_query(query, seed_planner)
                    evaluation[id] = seed_planner.evaluation.model_dump()

                    with open(evaluation_file, "w+") as fp:
                        json.dump(evaluation, fp)

                    progress_bar.update(1)
