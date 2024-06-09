import json
import re
from typing import Any, Dict, Optional

import session as SESSION
from car_state import CarState
from llm.base import LLMMessage
from plan.domain import (
    PlanFailureExecutorNotification,
    PlanRetryExecutorNotification,
    PlanSuccessExecutorNotification,
    Transition,
)
from plan.exceptions import MemoryException, ToolException, UnknownToolException
from plan.feedback import (
    ExecutorFailureFeedback,
    ExecutorOkFeedback,
    ExecutorRunBranch,
    ExecutorSkipBranch,
)
from plan.planner import LLMPlanner
from tool import tools


def _pre_process_args(fn_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if fn_name == "media_search":
        if "types" in args and isinstance(args["types"], (list, str)):
            if isinstance(args["types"], str):
                args["types"] = [args["types"]]
            args["types"] = tuple(args["types"])
        else:
            raise ToolException(
                tool_name="media_search",
                taxonomy="illegal_value",
                code=31,
                message="types parameter should be in args for "
                + "media_search and be a list",
            )
    return args


def replace_slots_with_memory(
    memory: Dict[str, Any],
    fn_args: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    def _replace_slots_subfn(val_with_slots: str) -> Optional[str]:
        matches = re.findall(r"\$(\w+)\$", val_with_slots)
        if len(matches) == 0:
            return val_with_slots
        for mt in matches:
            if mt.lower() not in memory:
                raise MemoryException(
                    taxonomy="illegal_read",
                    code=22,
                    message=f"Variable {mt} not found in memory",
                )
            recall_val = memory[mt.lower()]
            if isinstance(recall_val, (list, dict)):
                recall_val = json.dumps(recall_val)
            recall_val = str(recall_val)
            val_with_slots = val_with_slots.replace(f"${mt}$", recall_val)
        return val_with_slots

    if not fn_args:
        return {}
    for arg_name, arg_value in dict(fn_args).items():
        if arg_value is None:
            continue
        if not isinstance(arg_value, (dict, list, str)):
            continue
        if isinstance(arg_value, dict):
            arg_value = json.dumps(arg_value)
            arg_value = _replace_slots_subfn(arg_value)
            arg_value = json.loads(arg_value)
        elif isinstance(arg_value, list):
            arg_value = [_replace_slots_subfn(val) for val in arg_value]
            new_value = []
            # Avoid sublists when replacing slots - lists are kept one depth
            for sub_value in arg_value:
                try:
                    vals = json.loads(sub_value)
                    if isinstance(vals, list):
                        new_value = [*new_value, *vals]
                    if isinstance(vals, dict):
                        new_value.append(vals)
                except json.JSONDecodeError:
                    new_value.append(sub_value)
            arg_value = new_value
        else:
            arg_value = _replace_slots_subfn(arg_value)
        fn_args[arg_name] = arg_value
    return fn_args


def solve_query(query: str, planner: LLMPlanner) -> None:
    plan_iterator = planner.make_plan(query)
    finished = False
    while not finished:
        # Reset state for new try
        _memory = {}
        SESSION.CAR_STATE = CarState.get_default()
        # Run a plan of unknown number steps
        while True:
            step = next(plan_iterator)

            if isinstance(
                step, (PlanSuccessExecutorNotification, PlanFailureExecutorNotification)
            ):
                # Planner is finished with query
                finished = True

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

            if isinstance(step, list):
                # Assuming condition branch
                assert step[
                    0
                ].evaluate_condition, (
                    "Badly formed branch, no condition on first PlanStep"
                )
                response = SESSION.ORACLE.invoke(
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
                    planner.post_feedback(
                        ExecutorRunBranch(
                            tokens=response.tokens,
                            steps=step,
                        )
                    )
                    continue

                # Either false or default to no answer
                planner.post_feedback(
                    ExecutorSkipBranch(tokens=response.tokens),
                )
                continue

            # Simple tool step
            tool_name = step.tool_name
            if tool_name not in tools.TOOL_NAMES:
                planner.post_feedback(
                    ExecutorFailureFeedback(exception=UnknownToolException(code=23))
                )
                continue

            tool_args = step.args
            try:
                tool_args = replace_slots_with_memory(_memory, tool_args)
            except MemoryException as e:
                planner.post_feedback(ExecutorFailureFeedback(exception=e))
                continue

            try:
                tool_args = _pre_process_args(tool_name, tool_args)
                tool_output = getattr(tools, tool_name)(**tool_args)
            except ToolException as e:
                planner.post_feedback(ExecutorFailureFeedback(exception=e))
                continue
            except TypeError as e:
                planner.post_feedback(
                    ExecutorFailureFeedback(
                        exception=ToolException(
                            tool_name,
                            "wrong_signature",
                            code=27,
                            message=str(e),
                        )
                    )
                )
                continue

            # Save tool output to memory if required by tool
            if step.memory and tool_output:
                _memory[step.memory] = tool_output

            # Running the step went ok
            planner.post_feedback(
                ExecutorOkFeedback(
                    step=step,
                    tool_output=tool_output,
                    transition=Transition(
                        new_state=SESSION.CAR_STATE.model_copy(),
                        new_memory=dict(_memory),
                    ),
                )
            )

    assert planner.evaluation
