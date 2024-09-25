import json
import re
from typing import Any

import src.session as SESSION
from src.car_state import CarState
from src.llm import LLMMessage
from src.plan.domain import (
    PlanFailureExecutorNotification,
    PlanRetryExecutorNotification,
    PlanSuccessExecutorNotification,
    Transition,
)
from src.plan.exceptions import BenchmarkException, ExceptionCode
from src.plan.feedback import (
    ExecutorFailureFeedback,
    ExecutorOkFeedback,
    ExecutorRunBranch,
    ExecutorSkipBranch,
)
from src.plan.planner import PlannerInterface
from src.tool import tools


def _pre_process_args(fn_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if fn_name == "media_search":
        if "types" in args and isinstance(args["types"], (list, str)):
            if isinstance(args["types"], str):
                args["types"] = [args["types"]]
            args["types"] = tuple(args["types"])
        else:
            raise BenchmarkException(
                code=ExceptionCode.TOOL_SIGNATURE,
                message="media_search: `types` not present in args or is not a list",
            )
    return args


def replace_slots_with_memory(
    memory: dict[str, Any],
    fn_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Replace symbolic references to memory in function arguments with actual values."""

    def _replace_memory_reference(val_with_slots: str) -> str:
        matches = re.findall(r"\$(\w+)\$", val_with_slots)
        if len(matches) == 0:
            return val_with_slots
        for mt in matches:
            if mt.lower() not in memory:
                raise BenchmarkException(
                    code=ExceptionCode.MEMORY,
                    message=f"Variable {mt} not found in memory",
                )
            recall_val = memory[mt.lower()]
            if isinstance(recall_val, (list, dict)):
                recall_val = json.dumps(recall_val, ensure_ascii=False)
            if not isinstance(recall_val, str):
                # Do not encode strings, it will add extra quotes
                recall_val = str(recall_val)
            val_with_slots = val_with_slots.replace(f"${mt}$", recall_val)
            if val_with_slots[0] == '"' and val_with_slots[-1] == '"':
                # Remove quotes if present it messes up the json parsing
                val_with_slots = val_with_slots[1:-1]
        return val_with_slots

    if not fn_args:
        return {}
    for arg_name, arg_value in dict(fn_args).items():
        if arg_value is None:
            continue
        if not isinstance(arg_value, (dict, list, str)):
            continue
        if isinstance(arg_value, dict):
            arg_value_serialized = json.dumps(arg_value, ensure_ascii=False)
            arg_value_replaced = _replace_memory_reference(arg_value_serialized)
            # Remove quotes if present it messes up the json parsing
            if '""' in arg_value_replaced:
                arg_value_replaced = arg_value_replaced.replace('""', '"')
            arg_value = json.loads(arg_value_replaced)
        elif isinstance(arg_value, list):
            arg_value = [_replace_memory_reference(val) for val in arg_value]
            new_value = []
            # Avoid sublists when replacing slots - lists are kept one depth
            for sub_value in arg_value:
                try:
                    vals = json.loads(sub_value)
                    if isinstance(vals, str):
                        new_value.append(vals)
                    if isinstance(vals, list):
                        new_value = [*new_value, *vals]
                    if isinstance(vals, dict):
                        new_value.append(vals)
                except json.JSONDecodeError:
                    new_value.append(sub_value)
            arg_value = new_value
        else:
            arg_value = _replace_memory_reference(arg_value)
        fn_args[arg_name] = arg_value
    return fn_args


def solve_query(query: str, planner: PlannerInterface) -> None:
    """Given a user query, attempt to solve it using the planner."""
    plan_iterator = planner.make_plan(query)
    finished = False  # False until planner is done with query or gives up
    while not finished:
        # Reset state for new try
        _memory: dict[str, Any] = {}
        SESSION.CAR_STATE = CarState.get_default()
        # Run a plan of unknown number steps
        while True:
            # Get next step from plan
            step = next(plan_iterator)

            # Check if planner is done with query
            if isinstance(
                step, (PlanSuccessExecutorNotification, PlanFailureExecutorNotification)
            ):
                # Planner is finished with query
                finished = True

            # Give up current attempt at query
            if isinstance(
                step,
                (
                    PlanRetryExecutorNotification,
                    PlanSuccessExecutorNotification,
                    PlanFailureExecutorNotification,
                ),
            ):
                # Stop from current attempt at query
                break

            # Planner has requested to evaluate if a conditional branch should be taken
            # Both branches are returned in a list and evaluated
            if isinstance(step, list):
                # Assuming condition branch
                assert (
                    step[0].evaluate_condition
                ), "Badly formed branch, no condition on first PlanStep"
                assert SESSION.LLM
                response = SESSION.LLM.invoke(
                    [
                        LLMMessage(
                            role="assistant",
                            content=(
                                "Decide if statement is True or False. Output either "
                                "'true' or 'false' and nothing more"
                            ),
                        ),
                        LLMMessage(
                            role="user",
                            content=step[0].evaluate_condition,
                        ),
                    ]
                )
                if any(
                    truthy in response.text.lower()
                    for truthy in ["true", "yes", "indeed"]
                ):
                    # Inform planner to take the branch
                    planner.post_feedback(
                        ExecutorRunBranch(
                            tokens=response.tokens,
                            steps=step,
                        )
                    )
                    continue

                # The branch should be skipped and take the other branch
                planner.post_feedback(
                    ExecutorSkipBranch(tokens=response.tokens),
                )
                continue

            # Base case: try to run a tool call step
            tool_name = step.tool_name
            if tool_name not in tools.TOOL_NAMES:
                planner.post_feedback(
                    ExecutorFailureFeedback(
                        exception=BenchmarkException(
                            code=ExceptionCode.TOOL_SIGNATURE,
                            message=f"Tool {tool_name} not found in tools",
                        )
                    )
                )
                continue

            # Replace references to memory in tool arguments
            tool_args = step.args
            try:
                tool_args = replace_slots_with_memory(_memory, tool_args)
            except BenchmarkException as e:
                planner.post_feedback(ExecutorFailureFeedback(exception=e))
                continue

            # Call the tool
            try:
                tool_args = _pre_process_args(tool_name, tool_args)
                tool_output = getattr(tools, tool_name)(**tool_args)
            except BenchmarkException as e:
                # Forward the tool call fail to the planner
                planner.post_feedback(ExecutorFailureFeedback(exception=e))
                continue
            except TypeError as e:
                # Tool signature mismatch
                planner.post_feedback(
                    ExecutorFailureFeedback(
                        exception=BenchmarkException(
                            code=ExceptionCode.TOOL_SIGNATURE,
                            message=f"{tool_name}: {str(e)}",
                        )
                    )
                )
                continue

            # Save tool output to memory if required by tool
            if step.memory and tool_output:
                _memory[step.memory] = tool_output

            # Inform planner of successful tool call
            planner.post_feedback(
                ExecutorOkFeedback(
                    step=step,
                    tool_output=json.dumps(tool_output, ensure_ascii=False),
                    transition=Transition(
                        new_state=SESSION.CAR_STATE.model_copy(),
                        new_memory=dict(_memory),
                    ),
                )
            )

    # Assert the attempt at the query was registered
    # A record is produced for each attempt of the planner
    # at the query, successful or not
    assert planner.evaluation
