"""Notifications sent from planner to the strategies to inform about the current state of the plan."""

from pydantic import BaseModel, ConfigDict

from src.car_state import CarState
from src.plan.domain import PlanStep, Transition
from src.plan.exceptions import BenchmarkException


class NewQueryNotification(BaseModel):
    query: str
    alignment_state: CarState | None


class InstructionToExecuteNotification(BaseModel):
    step: PlanStep


class OkStrategyNotification(BaseModel):
    step: PlanStep
    tool_output: list | dict | str | None
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


StrategyNotification = (
    NewQueryNotification
    | OkStrategyNotification
    | ExceptionNotification
    | InstructionToExecuteNotification
    | PlanSuccessStrategyNotification
    | PlanRetryStrategyNotification
    | PlanFailureStrategyNotification
)
