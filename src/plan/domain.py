"""Step sent from planner to executor."""

import typing
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

if typing.TYPE_CHECKING:
    from src.car_state import CarState


class Transition(BaseModel):
    """Transition from one system state to another."""

    new_state: "CarState"
    new_memory: dict[str, Any]


class PlanStep(BaseModel):
    """Step sent from planner to executor."""

    evaluate_condition: str | None = None
    tool_name: str
    raw_plan_text: str | None = None
    args: dict[str, Any] | None = {}
    memory: str | None = None
    # Reason required for validation purposes only
    reason: str | None = None

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, tool_name):
        if not tool_name and not isinstance(tool_name, str):
            raise ValidationError("Expected tool_name field to be a string")
        return tool_name

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, memory):
        if memory is None:
            return memory
        if isinstance(memory, str):
            return memory.lower()
        raise ValidationError("Memory field could not be parsed, it should be a string")

    @field_validator("args")
    @classmethod
    def validate_args(cls, args: dict[str, Any] | None):
        if args is None:
            return {}
        if not isinstance(args, dict):
            raise ValidationError(f"Expected None or a dictionary: {args}")
        return args


class PlanRetryExecutorNotification(BaseModel):
    """Planner requests a retry of the plan from the executor.

    Executor will reset the state and retry the query with a new plan.
    """

    pass


class PlanSuccessExecutorNotification(BaseModel):
    """Executor notifies planner of successful plan execution.

    All steps of a plan have been executed successfully.
    """

    pass


class PlanFailureExecutorNotification(BaseModel):
    """Executor notifies planner of failed plan execution.

    Executor will not retry the query.
    """

    pass


PlannerOutput = (
    PlanStep
    | list[PlanStep]
    | PlanRetryExecutorNotification
    | PlanSuccessExecutorNotification
    | PlanFailureExecutorNotification
)


GeneratedPlanStep = list[PlanStep | list[PlanStep]]
