import json
import os
from pathlib import Path
import random
import re
from typing import Dict, List

import tqdm
from joblib import hash as joblib_hash

import session as SESSION
from dataset.read_seed import read_seed_dataset
from llm.openai import OpenAILLM
from plan.evaluation import QueryEvaluation
from plan.executor import solve_query
from plan.planner import SeedPlanner
import itertools


def _data_augmentation_gml_modes(plan: str, augment_count: int = 10) -> List[str]:
    all_nodes = re.findall(r'node \[.+?\]', plan)
    all_edges = re.findall(r'edge \[.+?\]', plan)
    
    all = [*all_nodes, *all_edges]
    random.shuffle(all)
    
    output = []
    for perm in itertools.islice(itertools.permutations(all), augment_count):
        output.append(
            f'graph [ directed 1 {" ".join(perm)}]'
        )
    
    return output


if __name__ == "__main__":
    LLM = OpenAILLM("gpt-3.5-turbo-0125")
    SESSION.ORACLE = LLM

    for mode in ["gml"]:
        seed_planner = SeedPlanner(LLM, mode)
        dataset = read_seed_dataset(mode)

        size = 0
        for query, seed_plans in dataset.items():
            if 'gml' in mode:
                # seed_plans = _data_augmentation_gml_modes(seed_plans[0])
                dataset[query] = seed_plans

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
