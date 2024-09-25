import json
from pathlib import Path


def read_seed_dataset(plan_format: str) -> dict[str, list[str]]:
    query_plan: dict[str, list[str]] = {}
    data_dir = (
        Path(__file__)
        .parent.parent.parent.joinpath("data")
        .joinpath("seed")
        .joinpath("datasets")
    )

    if plan_format in ("json", "json+r"):
        ds_path = {
            "json": data_dir.joinpath("seed.json"),
            "json+r": data_dir.joinpath("seed_reason.json"),
        }[plan_format]
        with open(ds_path, "r", encoding="utf-8") as fp:
            seed_dataset = json.load(fp)
        for example in seed_dataset:
            query = example["instruction"]
            query_plan[query] = []
            for plan in example["instances"]:
                query_plan[query].append(json.dumps(plan, ensure_ascii=False))

    if plan_format in ("gml", "gml+r", "gml+r+e"):
        ds_path = {
            "gml": data_dir.joinpath("seed_gml"),
            "gml+r": data_dir.joinpath("seed_gml_reason"),
            "gml+r+e": data_dir.joinpath("seed_gml_reason_plus_edges"),
        }[plan_format]
        with open(
            data_dir.joinpath("seed_gml"),
            "r",
            encoding="utf-8",
        ) as fp:
            for query in fp:
                plan = next(fp)
                query_plan[query] = [plan]

    return query_plan
