from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from car_state import CarState
from plan.domain import PlanStep, Transition
from plan.exceptions import BenchmarkException


class NewQueryNotification(BaseModel):
    query: str
    alignment_state: Optional[CarState]


class InstructionToExecuteNotification(BaseModel):
    instruction: PlanStep


class OkStrategyNotification(BaseModel):
    step: PlanStep
    tool_output: Union[List, Dict, str, None]
    transition: Transition


class ExceptionNotification(BaseModel):
    exception: BenchmarkException

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PlanSuccessStrategyNotification(BaseModel):
    pass


class PlanRetryStrategyNotification(BaseModel):
    pass


class PlanFailureStrategyNotification(BaseModel):
    pass


StrategyNotification = Union[
    NewQueryNotification,
    OkStrategyNotification,
    ExceptionNotification,
    InstructionToExecuteNotification,
    PlanSuccessStrategyNotification,
    PlanRetryStrategyNotification,
    PlanFailureStrategyNotification,
]
