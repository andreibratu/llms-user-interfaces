import itertools
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List

import tqdm
from joblib import hash as joblib_hash

import session as SESSION
from configuration import APP_CONFIG
from dataset.read_seed import read_seed_dataset
from llm.openai import OpenAILLM
from plan.evaluation import QueryEvaluation
from plan.executor import solve_query
from plan.planner import SeedPlanner


def _data_augmentation_gml_modes(gml_plan: str, augment_count: int = 10) -> List[str]:
    all_nodes = re.findall(r"node \[.+?\]", gml_plan)
    all_edges = re.findall(r"edge \[.+?\]", gml_plan)

    building_blocks = [*all_nodes, *all_edges]
    random.shuffle(building_blocks)

    output = []
    for perm in itertools.islice(
        itertools.permutations(building_blocks), augment_count
    ):
        output.append(f'graph [ directed 1 {" ".join(perm)}]')

    return output


if __name__ == "__main__":
    LLM = OpenAILLM("gpt-3.5-turbo-0125")
    SESSION.ORACLE = LLM

    for mode in APP_CONFIG.experiment.plan_output_mode:
        seed_planner = SeedPlanner(LLM, mode)
        dataset = read_seed_dataset(mode)

        # pylint: disable=invalid-name
        num_plans = 0
        for query, seed_plans in dataset.items():
            if "gml" in mode:
                # seed_plans = _data_augmentation_gml_modes(seed_plans[0])
                dataset[query] = seed_plans

            num_plans += len(seed_plans)

        evaluation_file = Path(__file__).parent.joinpath(
            "data", "seed_evaluate", f"{mode}_dataset_evaluation.json"
        )
        evaluation: Dict[str, QueryEvaluation] = {}
        os.makedirs(evaluation_file.parent, exist_ok=True)
        if evaluation_file.exists():
            with open(evaluation_file, "r", encoding="utf-8") as fp:
                evaluation: Dict[str, QueryEvaluation] = json.load(fp)

        with tqdm.tqdm(total=num_plans, desc=mode) as progress_bar:
            for query, seed_plans in dataset.items():

                for plan in seed_plans:
                    query_plan_id = joblib_hash(
                        f"{len(query)}{query}|{len(plan)}{plan}"
                    )

                    if query_plan_id in evaluation:
                        progress_bar.update(1)
                        continue

                    seed_planner.inject_query_plan(query, plan)
                    solve_query(query, seed_planner)
                    evaluation[query_plan_id] = seed_planner.evaluation.model_dump()

                    with open(evaluation_file, "w+", encoding="utf-8") as fp:
                        json.dump(evaluation, fp)

                    progress_bar.update(1)
