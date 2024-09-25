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

import nltk
import numpy as np
import tqdm
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

from src.configuration import APP_CONFIG
from src.llm import LLMInterface, LLMMessage, LLMResponse
from src.plan.exceptions import BenchmarkException
from src.tool.tools import TOOL_HEADERS

random.seed(APP_CONFIG.experiment.random_seed)


_SYSTEM_PROMPT = (
    "You are a car assistant. Invent scenarios user can ask you. "
    "Write them from user perspective. Try to include multiple tasks in each scenario. "
    "Use places of interest like zoos, museums, theatres or restaurants. "
    "Make them different from the example scenarios. "
    "Use kilometers and celsius degrees. "
    "Ask about local area. Only include scenarios that can be solved by available tools"
    f"You can control the car using these tools: {TOOL_HEADERS}. "
    "Come up with new scenario the assistant should be able to handle.\n"
)


def _generate_instructions_user_prompt(prompt_instructions: list[str]) -> str:
    user_prompt = "EXAMPLES\n"
    for idx, instruction in enumerate(prompt_instructions):
        user_prompt += f"{idx+1}. {instruction}\n"
    return user_prompt


def _sample_machine_instructions(
    machine_instructions: list[str], min_n: int
) -> list[str]:
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(
        machine_instructions,
        min(min_n, len(machine_instructions)),
    )


def _validate_instruction(text: str) -> bool:
    """Reject query generated for evaluation benchmark if needed.

    Reasons for rejection:
    - Query is too short or too long
    - Query starts with punctuation
    - Query contains non-ascii characters
    - Query does not end with punctuation
    - Query contains forbidden substrings
    - Query contains references to tools
    - Query contains only gerunds
    """
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
    # Reject generated query if contains a forbidden substring or a reference to a tool
    deny_list = APP_CONFIG.generation.deny_list + list(TOOL_HEADERS.keys())
    if any(token in text.lower() for token in deny_list):
        return False
    # Verbs check
    pos_categorization = [pos[1] for pos in nltk.pos_tag(word_tokenize(text))]
    verb_pos = [
        pos for pos in pos_categorization if pos in APP_CONFIG.generation.all_verbs
    ]
    if all(pos == "VBG" for pos in verb_pos):
        return False
    return True


def _process_llm_response(response: LLMResponse) -> list[str]:
    """
    Extract generated query from the response and validate it.
    """
    raw_instructions = re.split(r"\d+\s?\. ", response.text)
    queries = []
    for inst in raw_instructions:
        inside_quotes = re.match(r"[\"\'](.*)[\"\']", inst)
        if inside_quotes is not None:
            inst = inside_quotes[1]
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "" or not _validate_instruction(inst):
            continue
        queries.append(inst)
    return queries


def generate_benchmark_dataset(
    generator_llm: LLMInterface,
    seed_tasks_path: Path = Path("data", "seed", "datasets", "seed.json"),
    instructions_output_path: Path = Path("data", "benchmark", "benchmark.jsonl"),
    num_instructions_generate: int = 1000,
    num_prompt_instructions: int = 10,
    request_batch_size: int = 1,
):
    """Generate benchmark dataset for evaluation.

    Adaptation of Self-Instruct paper where synthetic instructions are generated
    to finetune the model on following instructions.
    https://arxiv.org/abs/2212.10560
    https://github.com/yizhongw/self-instruct

    Uses the rouge score to filter out generated instructions that are too
    similar to the seed tasks or other generations.

    Args:
        generator_llm: LLMInterface object.
        seed_tasks_path: Path to human annotated seed tasks.
        instructions_output_path: Path to save generated queries.
        num_instructions_generate: Number of queries to generate.
        num_prompt_instructions: Number of generated instructions to include in prompt as example.
            Set to 0 to only use queries from seed dataset.
        request_batch_size: Number of queries to generate in one request.
    """
    with open(seed_tasks_path, "r", encoding="utf-8") as fp:
        seed_data = json.load(fp)
    seed_queries = [inst["instruction"] for inst in seed_data]
    machine_queries = []

    if os.path.exists(instructions_output_path):
        with open(instructions_output_path, "r", encoding="utf-8") as fp:
            machine_queries = fp.readlines()

    scorer = rouge_scorer.RougeScorer(
        APP_CONFIG.generation.rouge_metrics,
        use_stemmer=False,
    )

    progress_bar = tqdm.tqdm(total=num_instructions_generate)
    if machine_queries:
        progress_bar.update(len(machine_queries))

    with open(instructions_output_path, "a", encoding="utf-8") as fout:
        old_time = None
        while len(machine_queries) < num_instructions_generate:
            curr_time = time.time()
            if old_time is not None:
                if curr_time - old_time > APP_CONFIG.generation.timeout_seconds:
                    # Cannot generate new instructions
                    break
            old_time = curr_time
            batch_user_prompts = []
            # Build the prompts for the batch
            for _ in range(request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = _sample_machine_instructions(
                    machine_queries,
                    min_n=APP_CONFIG.generation.min_machine_instructions_n,
                )
                # Sample
                prompt_instructions += random.sample(
                    seed_queries,
                    num_prompt_instructions - len(prompt_instructions),
                )
                random.shuffle(prompt_instructions)
                prompt = _generate_instructions_user_prompt(prompt_instructions)
                batch_user_prompts.append(prompt)

            # Generate queries
            try:
                results = [
                    generator_llm.invoke(
                        [
                            LLMMessage(
                                role="system",
                                content=_SYSTEM_PROMPT,
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
                # OpenAI fail, retry
                continue

            # Filter out empty responses
            results = [res for res in results if res and len(res.text) > 0]
            # Validate queries and filter those too similar
            new_queries, all_metadata = [], []
            for generated_instruction in results:
                valid_queries = _process_llm_response(generated_instruction)
                new_queries += valid_queries
                all_metadata += [res.metadata for res in results] * len(valid_queries)

            request_idx = 0
            for inst, metadata in zip(new_queries, all_metadata):
                # Parallelize the scoring of the generated instructions
                with ThreadPoolExecutor(8) as pool:
                    rouge_scores = pool.map(
                        partial(scorer.score, inst),
                        seed_queries + machine_queries,
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
                all_queries = seed_queries + machine_queries
                most_similar_instructions = {
                    all_queries[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                # Commit the generated queries to evaluation benchmark
                machine_queries.append(inst)
                fout.write(
                    json.dumps(
                        {
                            "instruction": inst,
                            "most_similar": most_similar_instructions,
                            "avg_similarity_score": float(np.mean(rouge_scores)),
                            "metadata": metadata,
                            "request_idx": request_idx,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                progress_bar.update(1)
            request_idx += 1
