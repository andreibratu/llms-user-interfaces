import json
import os
from pathlib import Path
import random
import time

from openai import OpenAI
from openai.types.fine_tuning import FineTuningJob
from openai.types import FileObject

from src.configuration import APP_CONFIG


random.seed(APP_CONFIG.experiment.random_seed)


STRATEGY = ["baseline", "tool_bert"]
PLAN_FORMAT = [
    # "json",
    "json+r",
    # "gml",
    # "gml+r",
    "gml+r+e",
]
FINETUNE_DIR = Path("data", "finetune")
OPENAI_CLIENT = OpenAI(api_key=APP_CONFIG.openai.api_key)


if os.path.exists("finetune_jobs.json"):
    with open("finetune_jobs.json", "r") as fp:
        finished_jobs = json.load(fp)
else:
    finished_jobs = {}


if os.path.exists("finetune_files.json"):
    with open("finetune_files.json", "r") as fp:
        finetune_files = json.load(fp)
else:
    finetune_files = {}


def _upload_finetune_file(s_name, p_format) -> str:
    if s_name == "baseline":
        dataset_fn = FINETUNE_DIR.joinpath(s_name, f"{p_format}.jsonl")
    else:
        all_examples = []
        ds_path = FINETUNE_DIR.joinpath(s_name, p_format)
        with open(ds_path.joinpath("fill_tool_op.jsonl"), "r") as fp:
            fill_tool_op = fp.readlines()
            if APP_CONFIG.experiment.finetune_tool_bert_fill_tool_count:
                all_examples.extend(
                    random.sample(
                        fill_tool_op,
                        APP_CONFIG.experiment.finetune_tool_bert_fill_tool_count,
                    )
                )
            else:
                all_examples.extend(fill_tool_op)
        for op_name in [
            "known_tools_unknown_end_state",
            "unknown_tools_known_end_state",
            "missing_tool_op",
        ]:
            with open(ds_path.joinpath(f"{op_name}.jsonl"), "r") as fp:
                op_rows = fp.readlines()
                all_examples.extend(
                    random.sample(
                        op_rows,
                        int(
                            APP_CONFIG.experiment.finetune_tool_bert_percentage
                            * len(op_rows)
                        ),
                    )
                )
        random.shuffle(all_examples)
        dataset_fn = ds_path.joinpath(
            (
                f"tool_bert_combined_{p_format}_"
                f"{APP_CONFIG.experiment.finetune_tool_bert_percentage}_"
                f"{APP_CONFIG.experiment.finetune_tool_bert_fill_tool_count}.jsonl"
            )
        )
        with open(dataset_fn, "w+") as fp:
            fp.writelines(all_examples)
    finetune_file: FileObject = OPENAI_CLIENT.files.create(
        file=open(dataset_fn, "rb"), purpose="fine-tune"
    )
    return finetune_file.id


for model in APP_CONFIG.experiment.openai_models:
    for s_name in STRATEGY:
        for p_format in PLAN_FORMAT:
            if s_name == "tool_bert":
                if p_format not in ["json+r", "gml+r+e"]:
                    # Tool_BERT is expensive to fine-tune
                    continue
            model_suffix = f"{s_name}_{p_format}"
            if model_suffix in finished_jobs:
                print(f"{model_suffix} is processed.")
                continue

            if model_suffix in finetune_files:
                finetune_file_id = finetune_files[model_suffix]
            else:
                finetune_file_id = _upload_finetune_file(s_name, p_format)
                finetune_files[model_suffix] = finetune_file_id
                with open("finetune_files.json", "w+") as fp:
                    json.dump(finetune_files, fp, indent=4)

            finetune_job: FineTuningJob = OPENAI_CLIENT.fine_tuning.jobs.create(
                model=model, training_file=finetune_file_id, suffix=model_suffix
            )
            print(f"Fine-tuning job {finetune_job.id} {model_suffix} is created.")

            while True:
                time.sleep(60)
                finetune_job = OPENAI_CLIENT.fine_tuning.jobs.retrieve(finetune_job.id)
                assert (
                    finetune_job.error is None
                ), f"Fine-tuning job {finetune_job.id} {model_suffix} failed."
                if finetune_job.finished_at is None:
                    print(
                        f"Fine-tuning job {finetune_job.id} {model_suffix} is still running."
                    )
                else:
                    break
            finished_jobs[model_suffix] = finetune_job.id
            with open("finetune_jobs.json", "w+") as fp:
                json.dump(finished_jobs, fp, indent=4)
