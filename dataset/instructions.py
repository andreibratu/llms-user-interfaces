import json
import os
import random
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from statistics import mean
from typing import List

import nltk
import numpy as np
import tqdm
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

from configuration import APP_CONFIG
from llm.base import LLMInterface, LLMMessage, LLMResponse
from plan.exceptions import BenchmarkException
from tool.tools import TOOL_HEADERS

random.seed(42)
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


def _generate_instructions_system_prompt() -> str:
    return (
        "You are a car assistant. Invent scenarios user can ask you. "
        "Write them from user perspective. Try to include multiple tasks in each scenario. "
        "Use places of interest like zoos, museums, theatres or restaurants. "
        "Make them different from the example scenarios. "
        "Use kilometers and celsius degrees. "
        "Ask about local area. Only include scenarios that can be solved by available tools"
        f"You can control the car using these tools: {TOOL_HEADERS}. "
        "Come up with new scenario the assistant should be able to handle.\n"
    )


def _generate_instructions_user_prompt(prompt_instructions: List[str]) -> str:
    user_prompt = "EXAMPLES\n"
    for idx, instruction in enumerate(prompt_instructions):
        user_prompt += f"{idx+1}. {instruction}\n"
    return user_prompt


def _sample_machine_instructions(
    machine_instructions: List[str], min_n: int
) -> List[str]:
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(
        machine_instructions,
        min(min_n, len(machine_instructions)),
    )


def _validate_instruction(text: str) -> bool:
    # Simple checks
    if any(
        [
            len(text) <= APP_CONFIG.generation.min_len
            or len(text) > APP_CONFIG.generation.max_len,
            text[0] in string.punctuation.capitalize(),
            any(not cr.isascii() for cr in text),
            text[-1] not in [".", "?"],  # Ill-formed query, likely not complete
        ]
    ):
        return False
    # Deny-list check
    deny_list = APP_CONFIG.generation.deny_list + list(TOOL_HEADERS.keys())
    if any(token in text.lower() for token in deny_list):
        return False
    # Verbs check
    pos_categorisation = [pos[1] for pos in nltk.pos_tag(word_tokenize(text))]
    verb_pos = [
        pos for pos in pos_categorisation if pos in APP_CONFIG.generation.all_verbs
    ]
    if all(pos == "VBG" for pos in verb_pos):
        return False
    return True


def _process_oracle_response(response: LLMResponse) -> List[str]:
    raw_instructions = re.split(r"\d+\s?\. ", response.text)
    instructions = []
    for inst in raw_instructions:
        inside_quotes = re.match(r"[\"\'](.*)[\"\']", inst)
        if inside_quotes is not None:
            inst = inside_quotes[1]
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "" or not _validate_instruction(inst):
            continue
        instructions.append(inst)
    return instructions


# pylint: disable=too-many-arguments,too-many-branches
def generate_instructions(
    generator_llm: LLMInterface,
    seed_tasks_path: Path = Path("data", "seed.json"),
    instructions_output_path: Path = Path("data", "generated.jsonl"),
    num_instructions_generate: int = 1000,
    num_prompt_instructions: int = 10,
    request_batch_size: int = 1,
):
    with open(seed_tasks_path, "r", encoding="utf-8") as fp:
        seed_data = json.load(fp)
    seed_instructions = [inst["instruction"] for inst in seed_data]
    machine_instructions = []

    if os.path.exists(instructions_output_path):
        with open(instructions_output_path, "r", encoding="utf-8") as fp:
            machine_instructions = fp.readlines()

    scorer = rouge_scorer.RougeScorer(
        APP_CONFIG.generation.rouge_metrics,
        use_stemmer=False,
    )

    progress_bar = tqdm.tqdm(total=num_instructions_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(instructions_output_path, "a", encoding="utf-8") as fout:
        old_time = None
        while len(machine_instructions) < num_instructions_generate:
            curr_time = time.time()
            if old_time is not None:
                if curr_time - old_time > APP_CONFIG.generation.timeout_seconds:
                    print("Generation too slow, stopping")
                    break
            old_time = curr_time
            batch_user_prompts = []
            for _ in range(request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = _sample_machine_instructions(
                    machine_instructions, min_n=2
                )
                # sample human instructions from the pool
                prompt_instructions += random.sample(
                    seed_instructions,
                    num_prompt_instructions - len(prompt_instructions),
                )
                random.shuffle(prompt_instructions)
                prompt = _generate_instructions_user_prompt(prompt_instructions)
                batch_user_prompts.append(prompt)

            try:
                results = [
                    generator_llm.invoke(
                        [
                            LLMMessage(
                                role="system",
                                content=_generate_instructions_system_prompt(),
                            ),
                            LLMMessage(role="user", content=user_prompt),
                        ],
                        max_tokens=APP_CONFIG.generation.max_tokens,
                        temperature=APP_CONFIG.generation.temperature,
                        frequency_penalty=APP_CONFIG.generation.frequency_penalty,
                        presence_penalty=APP_CONFIG.generation.presence_penalty,
                        top_p=APP_CONFIG.generation.top_p,
                        n=APP_CONFIG.generation.n,
                    )
                    for user_prompt in batch_user_prompts
                ]
            except BenchmarkException:
                # TODO: Handle filtering policy more elegantly
                continue
            results = [res for res in results if res and len(res.text) > 0]
            instructions, all_metadata = [], []
            for generated_instruction in results:
                new_instructions = _process_oracle_response(generated_instruction)
                instructions += new_instructions
                all_metadata += [res.metadata for res in results] * len(
                    new_instructions
                )

            request_idx = 0
            for inst, metadata in zip(instructions, all_metadata):
                with ThreadPoolExecutor(8) as pool:
                    rouge_scores = pool.map(
                        partial(scorer.score, inst),
                        seed_instructions + machine_instructions,
                    )
                rouge_scores = [
                    mean(
                        score[rouge_t].fmeasure
                        for rouge_t in APP_CONFIG.generation.rouge_metrics
                    )
                    for score in rouge_scores
                ]
                if max(rouge_scores) > APP_CONFIG.generation.rouge_threshold:
                    continue
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                machine_instructions.append(inst)
                fout.write(
                    json.dumps(
                        {
                            "instruction": inst,
                            "most_similar": most_similar_instructions,
                            "avg_similarity_score": float(np.mean(rouge_scores)),
                            "metadata": metadata,
                            "request_idx": request_idx,
                        }
                    )
                    + "\n"
                )
                progress_bar.update(1)
            request_idx += 1
