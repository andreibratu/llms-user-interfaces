import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from configuration import APP_CONFIG
from llm.base import LLMMessage
from prompts import get_system_prompt


seed_evaluations_dir = Path(__file__).parent.joinpath("data", "seed_evaluate")
finetune_dir = Path(__file__).parent.joinpath("data", "finetune")
simple_finetune_dir = finetune_dir.joinpath("simple")
tool_bert_finetune_dir = finetune_dir.joinpath("tool_bert")


def _build_simple_finetuning_dataset(plan_mode: str, dataset: Dict) -> List:
    finetune_dataset = []

    for _, query_eval in dataset.items():
        query = query_eval["query"]
        attempt = query_eval["attempts"][0]
        llm_plan_output = attempt["raw_llm_text"][0]
        finetune_dataset.append(
            {
                "messages": [
                    (
                        LLMMessage(
                            role="system", content=get_system_prompt(plan_mode)
                        ).model_dump(),
                        LLMMessage(role="user", content=query).model_dump(),
                        LLMMessage(
                            role="assistant", content=llm_plan_output
                        ).model_dump(),
                    )
                ]
            }
        )

    mode_sft_path = simple_finetune_dir.joinpath(f"{plan_mode}.jsonl")
    with open(mode_sft_path, "w", encoding="utf-8") as fp:
        for row in finetune_dataset:
            fp.write(f"{json.dumps(row)}\n")


def _unknown_tools_known_end_state_op(
    collection: List,
    query,
    plan_mode,
    tool_slice,
    start_car_state,
    start_memory,
    end_car_state,
    end_memory,
    transient_tool_outputs,
):
    messages = [
        LLMMessage(role="system", content=get_system_prompt(plan_mode)).model_dump(),
        LLMMessage(
            role="user", content=f"You must solve user query: {query}"
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                f"Initial car state: {json.dumps(start_car_state)}; "
                f"initial memory {json.dumps(start_memory)}"
            ),
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                "What tools must be used to achieve car state "
                f"{json.dumps(end_car_state)} and memory {json.dumps(end_memory)}"
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


def _known_tools_missing_end_state_op(
    collection: List,
    query,
    plan_mode,
    tool_slice,
    start_car_state,
    start_memory,
    end_car_state,
    end_memory,
    transient_tool_outputs,
):
    messages = [
        LLMMessage(role="system", content=get_system_prompt(plan_mode)).model_dump(),
        LLMMessage(
            role="user", content=f"You must solve user query: {query}"
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                f"Initial car state: {json.dumps(start_car_state)}; "
                f"initial memory {json.dumps(start_memory)}"
            ),
        ).model_dump(),
        LLMMessage(
            role="user",
            content=f"What is the car state and memory after using tools {tool_slice}",
        ).model_dump(),
        LLMMessage(
            role="assistant",
            content=f"Car state: {json.dumps(end_car_state)}; Memory: {json.dumps(end_memory)}",
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


def _missing_tool_op(
    collection: List,
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
    messages = [
        LLMMessage(role="system", content=get_system_prompt(plan_mode)).model_dump(),
        LLMMessage(
            role="user", content=f"You must solve user query: {query}"
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                f"Initial car state: {json.dumps(start_car_state)}; "
                f"initial memory {json.dumps(start_memory)}"
            ),
        ).model_dump(),
        LLMMessage(
            role="user",
            content=(
                f"Desired end state is {json.dumps(end_car_state)}; "
                f"final memory {json.dumps(end_memory)}"
            ),
        ).model_dump(),
        LLMMessage(
            role="user",
            content=f"The tool sequence is {tools_before} <BLANK> {tools_after}",
        ).model_dump(),
        LLMMessage(
            role="user", content="What tool should be used instead of <BLANK>?"
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


def _fill_tool_op(collection: List, tool_text: str, plan_mode: str):

    def split_fn_name_args(fn_call_text: str) -> Tuple[str, str]:
        if "json" in plan_mode:
            fn_call = json.loads(fn_call_text)
            return fn_call["tool_name"], fn_call["args"]
        args = {}
        args_mt = re.search(r"node \[ id \d+ function \"(.+)\" (.*) *\]", tool_text)
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

    fn_name, args = split_fn_name_args(tool_text)

    collection.append(
        {
            "messages": [
                LLMMessage(
                    role="system", content=get_system_prompt(plan_mode)
                ).model_dump(),
                LLMMessage(
                    role="user",
                    content=f"How do you call function {fn_name} with args {args}?",
                ).model_dump(),
                LLMMessage(role="assistant", content=tool_text).model_dump(),
            ]
        }
    )


def _format_tools_for_prompt(tools: List, plan_mode: str, full_plan_text: str) -> str:
    if "json" in plan_mode:
        return " ".join(json.dumps(tool) for tool in tools)
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


def _build_tool_bert_finetuning_dataset(plan_mode: str, dataset: Dict):
    known_tools_unknown_end_state = []
    unknwon_tools_known_end_state = []
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
                        tool["raw_plan_text"] = json.dumps(tool_cpy)

                tool_slice = [tool["raw_plan_text"] for tool in tools[i : j + 1]]

                assert len(tool_slice) > 0, "Tool slice should not be empty"
                assert all(
                    tool["raw_plan_text"] is not None for tool in tools[i : j + 1]
                ), "All tools should have raw_plan_text set"

                if len(tool_slice) == 1:
                    _fill_tool_op(fill_tool_op, tool_slice[0], plan_mode)

                start_car_state = (
                    init_car_state if i == 0 else transition_car_states[i - 1]
                )
                end_car_state = transition_car_states[j]

                start_memory = init_memory if i == 0 else transition_memory[i - 1]
                end_memory = transition_memory[j]

                _unknown_tools_known_end_state_op(
                    unknwon_tools_known_end_state,
                    query,
                    plan_mode,
                    _format_tools_for_prompt(tool_slice, plan_mode, full_plan_text),
                    start_car_state,
                    start_memory,
                    end_car_state,
                    end_memory,
                    transient_tool_outputs,
                )

                _known_tools_missing_end_state_op(
                    known_tools_unknown_end_state,
                    query,
                    plan_mode,
                    _format_tools_for_prompt(tool_slice, plan_mode, full_plan_text),
                    start_car_state,
                    start_memory,
                    end_car_state,
                    end_memory,
                    transient_tool_outputs,
                )

                for k in range(1, len(tool_slice) - 1):
                    tools_before = tool_slice[:k]
                    tools_after = tool_slice[k + 1 :]
                    missing_tool = tool_slice[k]

                    _missing_tool_op(
                        missing_tool_op,
                        query,
                        plan_mode,
                        _format_tools_for_prompt(
                            tools_before, plan_mode, full_plan_text
                        ),
                        _format_tools_for_prompt(
                            tools_after, plan_mode, full_plan_text
                        ),
                        _format_tools_for_prompt(
                            [missing_tool], plan_mode, full_plan_text
                        ),
                        start_car_state,
                        start_memory,
                        end_car_state,
                        end_memory,
                        transient_tool_outputs,
                    )

        mode_tool_bert_dir_path = tool_bert_finetune_dir.joinpath(plan_mode)
        os.makedirs(mode_tool_bert_dir_path, exist_ok=True)
        for op_name, op_collection in [
            ("known_tools_unknown_end_state", known_tools_unknown_end_state),
            ("unknown_tools_known_end_state", unknwon_tools_known_end_state),
            ("missing_tool_op", missing_tool_op),
            ("fill_tool_op", fill_tool_op),
        ]:
            with open(
                mode_tool_bert_dir_path.joinpath(f"{op_name}.jsonl"),
                "w+",
                encoding="utf-8",
            ) as fp:
                for row in op_collection:
                    fp.write(f"{json.dumps(row)}\n")


if __name__ == "__main__":
    assert (
        seed_evaluations_dir.exists()
    ), f"{seed_evaluations_dir} does not exist. Have you run evaluate_seed first?"

    os.makedirs(finetune_dir, exist_ok=True)
    os.makedirs(simple_finetune_dir, exist_ok=True)
    os.makedirs(tool_bert_finetune_dir, exist_ok=True)

    for mode in APP_CONFIG.experiment.plan_output_mode:
        seed_mode_ds = seed_evaluations_dir.joinpath(f"{mode}.json")

        with open(seed_mode_ds, "r", encoding="utf-8") as fp:
            ds = json.load(fp)

        _build_simple_finetuning_dataset(mode, ds)

        _build_tool_bert_finetuning_dataset(mode, ds)
