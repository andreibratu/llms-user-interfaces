import json
import os
import re
from pathlib import Path
from typing import Any

from src.configuration import APP_CONFIG
from src.domain import PlanFormat
from src.llm import LLMMessage
from src.plan.evaluation import QueryEvaluation
from src.prompts import get_system_message

SEED_EVALUATIONS_DIR = Path("data", "seed", "evaluation")
FINETUNE_DIR = Path("data", "finetuning")
BASELINE_FINETUNE_DIR = FINETUNE_DIR.joinpath("baseline")
TOOL_BERT_FINETUNE_DIR = FINETUNE_DIR.joinpath("tool_bert")


def build_finetuning_datasets():
    assert (
        SEED_EVALUATIONS_DIR.exists()
    ), f"{SEED_EVALUATIONS_DIR} does not exist. Have you run evaluate_seed first?"

    os.makedirs(FINETUNE_DIR, exist_ok=True)
    os.makedirs(BASELINE_FINETUNE_DIR, exist_ok=True)
    os.makedirs(TOOL_BERT_FINETUNE_DIR, exist_ok=True)

    for plan_format in APP_CONFIG.experiment.plan_formats:
        evaluated_seed_dataset_file = SEED_EVALUATIONS_DIR.joinpath(
            f"{plan_format}.json"
        )

        with open(evaluated_seed_dataset_file, "r", encoding="utf-8") as fp:
            evaluated_seed_dataset = json.load(fp)

        _build_baseline_finetuning_dataset(
            plan_format,
            evaluated_seed_dataset,
        )

        _build_tool_bert_finetuning_dataset(
            plan_format,
            evaluated_seed_dataset,
        )


def _build_baseline_finetuning_dataset(
    plan_format: PlanFormat, dataset: dict[str, dict]
):
    """Build a baseline finetuning dataset for a given plan mode.

    The baseline approach fine-tunes the model on tuples of
    (query, human_expert_plan). The human annotated plan is found in
    the raw_llm_text, refer to src.plan.planner.SeedPlanner.
    """
    finetune_dataset = []

    for _, query_eval in dataset.items():
        parsed_query_eval = QueryEvaluation.model_validate(query_eval)
        query = parsed_query_eval.query
        attempt = parsed_query_eval.attempts[0]
        llm_plan_output = json.dumps(attempt.raw_llm_text)
        finetune_dataset.append(
            {
                "messages": [
                    get_system_message(plan_format).model_dump(),
                    LLMMessage(role="user", content=query).model_dump(),
                    LLMMessage(role="assistant", content=llm_plan_output).model_dump(),
                ]
            }
        )

    mode_sft_path = BASELINE_FINETUNE_DIR.joinpath(f"{plan_format}.jsonl")
    with open(mode_sft_path, "w", encoding="utf-8") as fp:
        print("baseline", plan_format, len(finetune_dataset))
        for row in finetune_dataset:
            fp.write(f"{json.dumps(row, separators=(',', ':'), ensure_ascii=False)}\n")


def _build_tool_bert_finetuning_dataset(plan_format: str, dataset: dict):
    """Thesis proposal for a specialized finetuning procedure for LLMs.

    Called ToolBERT, in homage to how BERT embeddings are trained by masking
    tokens and predicting them given the context around. ToolBERT uses three
    operations to generated the finetuning dataset:
        1. Known tools with unknown end state - Given a sequence of tools and a start
            state for the system, predict the end state of the system.
        2. Unknown tools with known end state - Given a start and end state of the system,
            predict the tools used to transition the system.
        3. Missing tool - Given a sequence of tools, the start state and the end state of
            the system, predict the missing tool in the sequence.

    """
    known_tools_unknown_end_state = []
    unknown_tools_known_end_state = []
    missing_tool_op = []
    fill_tool_op = []

    for _, query_eval in dataset.items():
        query = query_eval["query"]
        attempt = query_eval["attempts"][0]
        full_plan_text = attempt["raw_llm_text"][0]
        transient_tool_outputs = attempt["transient_tool_outputs"]
        init_car_state = attempt["car_states"][0]
        init_memory = attempt["memory_states"][0]
        transition_car_states = attempt["car_states"][1:]
        transition_memory = attempt["memory_states"][1:]
        tools = attempt["executed_plan"]
        assert (
            len(tools) == len(transition_car_states) == len(transition_memory)
        ), "Length of tools, car_states and memory_states should be same"

        for i in range(len(tools)):
            for j in range(i, len(tools)):
                for tool in tools[i : j + 1]:
                    if tool["raw_plan_text"] is None:
                        tool_cpy = dict(tool)
                        del tool_cpy["raw_plan_text"]
                        del tool_cpy["evaluate_condition"]
                        tool["raw_plan_text"] = json.dumps(
                            tool_cpy,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )

                tool_slice = [tool["raw_plan_text"] for tool in tools[i : j + 1]]
                if len(tool_slice) > APP_CONFIG.experiment.max_tool_slice_size:
                    # Context window might be too large
                    continue

                assert len(tool_slice) > 0, "Tool slice should not be empty"
                assert all(
                    tool["raw_plan_text"] is not None for tool in tools[i : j + 1]
                ), "All tools should have raw_plan_text set"

                if len(tool_slice) == 1:
                    _fill_tool_operation(
                        collection=fill_tool_op,
                        tool_text=tool_slice[0],
                        plan_format=plan_format,
                    )

                start_car_state = (
                    init_car_state if i == 0 else transition_car_states[i - 1]
                )
                end_car_state = transition_car_states[j]

                start_memory = init_memory if i == 0 else transition_memory[i - 1]
                end_memory = transition_memory[j]

                _unknown_tools_known_end_state_operation(
                    collection=unknown_tools_known_end_state,
                    query=query,
                    plan_format=plan_format,
                    tool_slice=_format_tools_for_prompt(
                        tool_slice, plan_format, full_plan_text
                    ),
                    start_car_state=start_car_state,
                    start_memory=start_memory,
                    end_car_state=end_car_state,
                    transient_tool_outputs=transient_tool_outputs,
                )

                _known_tools_missing_end_state_operation(
                    collection=known_tools_unknown_end_state,
                    query=query,
                    plan_format=plan_format,
                    tool_slice=_format_tools_for_prompt(
                        tool_slice, plan_format, full_plan_text
                    ),
                    start_car_state=start_car_state,
                    start_memory=start_memory,
                    end_car_state=end_car_state,
                    end_memory=end_memory,
                    transient_tool_outputs=transient_tool_outputs,
                )

                for k in range(1, len(tool_slice) - 1):
                    tools_before = tool_slice[:k]
                    tools_after = tool_slice[k + 1 :]
                    missing_tool = tool_slice[k]

                    _missing_tool_operation(
                        missing_tool_op,
                        query,
                        plan_format,
                        _format_tools_for_prompt(
                            tools_before, plan_format, full_plan_text
                        ),
                        _format_tools_for_prompt(
                            tools_after, plan_format, full_plan_text
                        ),
                        _format_tools_for_prompt(
                            [missing_tool], plan_format, full_plan_text
                        ),
                        start_car_state,
                        start_memory,
                        end_car_state,
                        end_memory,
                        transient_tool_outputs,
                    )

    tool_bert_plan_format_file = TOOL_BERT_FINETUNE_DIR.joinpath(plan_format)
    os.makedirs(tool_bert_plan_format_file, exist_ok=True)
    for op_name, op_collection in [
        ("known_tools_unknown_end_state", known_tools_unknown_end_state),
        ("unknown_tools_known_end_state", unknown_tools_known_end_state),
        ("missing_tool_op", missing_tool_op),
        ("fill_tool_op", fill_tool_op),
    ]:
        with open(
            tool_bert_plan_format_file.joinpath(f"{op_name}.jsonl"),
            "w+",
            encoding="utf-8",
        ) as fp:
            for row in op_collection:
                fp.write(
                    f"{json.dumps(row, separators=(',', ':'), ensure_ascii=False)}\n"
                )


def _unknown_tools_known_end_state_operation(
    collection: list,
    query,
    plan_format,
    tool_slice,
    start_car_state,
    start_memory,
    end_car_state,
    transient_tool_outputs,
):
    changes = []
    for k in start_car_state.keys():
        if end_car_state[k] != start_car_state[k]:
            changes.append(k)
    if len(changes) == 0:
        return
    messages = [
        get_system_message(plan_format).model_dump(),
        LLMMessage(
            role="user", content=f"You must solve user query: {query}"
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                f"Car state: {json.dumps(start_car_state, separators=(',', ':'), ensure_ascii=False)}; "
                f"Memory {json.dumps(start_memory, separators=(',', ':'), ensure_ascii=False)}"
            ),
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                "How do you change the following properties:\n"
                "key: initial_value -> final_value\n"
                + "\n".join(
                    f"{k}: {start_car_state[k]} -> {end_car_state[k]}" for k in changes
                )
            ),
        ).model_dump(),
        LLMMessage(role="assistant", content=f"{tool_slice}").model_dump(),
    ]
    if transient_tool_outputs:
        messages.insert(
            1,
            LLMMessage(
                role="user",
                content=f"Conditions outside the car are: {transient_tool_outputs}",
            ).model_dump(),
        )
    collection.append({"messages": messages})


def _known_tools_missing_end_state_operation(
    collection: list,
    query,
    plan_format,
    tool_slice,
    start_car_state,
    start_memory,
    end_car_state,
    end_memory,
    transient_tool_outputs,
):
    changes = []
    for k in start_car_state.keys():
        if end_car_state[k] != start_car_state[k]:
            changes.append(k)
    if len(changes) == 0:
        return
    messages = [
        get_system_message(plan_format).model_dump(),
        LLMMessage(
            role="user", content=f"You must solve user query: {query}"
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                f"Car state: {json.dumps(start_car_state, separators=(',', ':'), ensure_ascii=False)}; "
                f"Memory {json.dumps(start_memory, separators=(',', ':'), ensure_ascii=False)}"
            ),
        ).model_dump(),
        LLMMessage(
            role="user",
            content=f"What is car state after running {tool_slice}. Use 'key: old_value -> new_value' format",
        ).model_dump(),
        LLMMessage(
            role="assistant",
            content=json.dumps(
                "\n".join(
                    [f"{k} {start_car_state[k]} -> {end_car_state[k]}" for k in changes]
                ),
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        ).model_dump(),
        LLMMessage(role="user", content="How does memory change?").model_dump(),
        LLMMessage(
            role="assistant",
            content=json.dumps(
                end_memory,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        ).model_dump(),
    ]
    if transient_tool_outputs:
        messages.insert(
            1,
            LLMMessage(
                role="user",
                content=f"Conditions outside the car are: {transient_tool_outputs}",
            ).model_dump(),
        )
    collection.append({"messages": messages})


def _missing_tool_operation(
    collection: list,
    query,
    plan_mode,
    tools_before,
    tools_after,
    missing_tool,
    start_car_state,
    start_memory,
    end_car_state,
    end_memory,
    transient_tool_outputs,
):
    changes = []
    for k in start_car_state.keys():
        if end_car_state[k] != start_car_state[k]:
            changes.append(k)
    if len(changes) == 0:
        return
    messages = [
        get_system_message(plan_mode).model_dump(),
        LLMMessage(
            role="user", content=f"You must solve user query: {query}"
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                f"Car state: {json.dumps(start_car_state, separators=(',', ':'), ensure_ascii=False)}; "
                f"Memory {json.dumps(start_memory, separators=(',', ':'), ensure_ascii=False)}"
            ),
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                "The following properties have changed after calling tools:\n"
                + "\n".join(
                    f"{k}: {start_car_state[k]} -> {end_car_state[k]}" for k in changes
                )
            ),
        ).model_dump(),
        LLMMessage(
            role="user",
            content=f"Memory after calling tools: {json.dumps(end_memory, separators=(',', ':'), ensure_ascii=False)}",
        ).model_dump(),
        LLMMessage(
            role="user",
            content=f"Fill the blank with a tool: {tools_before} <BLANK> {tools_after}",
        ).model_dump(),
        LLMMessage(role="assistant", content=missing_tool).model_dump(),
    ]
    if transient_tool_outputs:
        messages.insert(
            1,
            LLMMessage(
                role="user",
                content=f"Conditions outside the car are: {transient_tool_outputs}",
            ).model_dump(),
        )
    collection.append({"messages": messages})


def _fill_tool_operation(collection: list, tool_text: str, plan_format: str):
    def split_fn_name_args(fn_call_text: str) -> tuple[str, dict[str, Any]] | None:
        if "condition" in fn_call_text:
            # Conditional branch
            return None
        if "json" in plan_format:
            fn_call = json.loads(fn_call_text)
            return fn_call["tool_name"], fn_call["args"]
        args = {}
        args_mt = re.search(r"node \[ id \d+ function \"(.+)\" (.*) *\]", tool_text)
        assert args_mt, "Tool text should have function name and args"
        fn_name = args_mt[1]
        if args_mt[2]:
            # We also have args
            splits = args_mt[2].strip().split(" ")
            assert (
                len(splits) % 2 == 0
            ), "We should have pairs of keys and values in args"
            for i in range(0, len(splits), 2):
                key = splits[i]
                if key == "reason":
                    # Need arguments only
                    continue
                value = splits[i + 1]
                if value.find('"') != -1:
                    # Strip quotes from string value
                    value = value[1:-1]
                args[key] = value
        return fn_name, args

    if split_fn_name_args(tool_text) is None:
        # Skip conditional branches
        return

    fn_name, args = split_fn_name_args(tool_text)

    collection.append(
        {
            "messages": [
                get_system_message(plan_format).model_dump(),
                LLMMessage(
                    role="user",
                    content=f"How do you call function {fn_name} with args {args}?",
                ).model_dump(),
                LLMMessage(role="assistant", content=tool_text).model_dump(),
            ]
        }
    )


def _format_tools_for_prompt(tools: list, plan_mode: str, full_plan_text: str) -> str:
    if "json" in plan_mode:
        return " ".join(
            json.dumps(tool, separators=(",", ":"), ensure_ascii=False)
            for tool in tools
        )
    # GML plan, add relevant edges
    ids = set()
    for node in tools:
        mt = re.search(r"\[ id (\d+) .+? \]", node)
        assert mt, "Ill-formed GML node"
        ids.add(mt.group(1))
    edges = []
    for mt in re.findall(
        r"(edge \[ source (\d+) target (\d+)[\w\s\"]*? \])", full_plan_text
    ):
        if mt[1] in ids or mt[2] in ids:
            edges.append(mt[0])
    return " ".join(tools + edges)
