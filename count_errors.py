from collections import Counter
import json
from pathlib import Path
import pandas as pd


with open(Path("data", "benchmark", "errors.json"), "r", encoding="utf-8") as fp:
    errors = json.load(fp)
    grouped_errors = {}
    identifier_sets = {}

    for error in errors:
        key = (error["plan_format"], error["finetuning_strategy"])
        if key not in grouped_errors:
            grouped_errors[key] = []
            identifier_sets[key] = set()
        grouped_errors[key].append(error["code"])
        identifier_sets[key].add(error["identifier"])

    # Convert grouped_errors to a DataFrame
    data = []
    for key, codes in grouped_errors.items():
        counter = Counter(codes)
        for code, count in counter.items():
            average_count = count / len(identifier_sets[key])
            data.append([key[0], key[1], code, average_count])

    df = pd.DataFrame(
        data,
        columns=["Plan Format", "Finetuning Strategy", "Error Code", "Average Count"],
    )

    # Save the DataFrame to a CSV file
    df.to_csv(
        Path("data", "benchmark", "grouped_errors.csv"), index=False, encoding="utf-8"
    )

    for key, codes in grouped_errors.items():
        print(f"{key}: {Counter(codes)}")
