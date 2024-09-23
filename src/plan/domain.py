"""Instructions sent from planner to executor."""

from typing import Any, Optional, dict

from pydantic import BaseModel, ValidationError, field_validator

from src.car_state import CarState


class Transition(BaseModel):
    new_state: CarState
    new_memory: dict[str, Any]


class PlanStep(BaseModel):
    evaluate_condition: Optional[str] = None
    tool_name: str
    raw_plan_text: Optional[str] = None
    args: Optional[dict[str, Any]] = {}
    memory: Optional[str] = None
    # Reason required for validation purposes only
    reason: Optional[str] = None

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
    def validate_args(cls, args: Optional[dict[str, Any]]):
        if args is None:
            return {}
        if not isinstance(args, dict):
            raise ValidationError(f"Expected None or a dictionary: {args}")
        # for arg_name, arg_val in args.items():
        #     if isinstance(arg_val, (list, dict)):
        #         # Keep arguments simple, LLM should parse them
        #         args[arg_name] = json.dumps(arg_val)
        return args


class PlanRetryExecutorNotification(BaseModel):
    pass


class PlanSuccessExecutorNotification(BaseModel):
    pass


class PlanFailureExecutorNotification(BaseModel):
    pass


PlannerOutput = (
    PlanStep
    | list[PlanStep]
    | PlanRetryExecutorNotification
    | PlanSuccessExecutorNotification
    | PlanFailureExecutorNotification
)


PlanType = list[PlanStep | list[PlanStep]]
