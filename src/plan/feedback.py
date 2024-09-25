"""Notifications sent from executor to planner."""

from pydantic import BaseModel, ConfigDict

from src.plan.domain import PlanStep, Transition
from src.plan.exceptions import BenchmarkException


class ExecutorOkFeedback(BaseModel):
    step: PlanStep
    tool_output: str
    transition: Transition


class ExecutorRunBranch(BaseModel):
    tokens: int  # Tokens used to decide the conditional
    steps: list[PlanStep]


class ExecutorSkipBranch(BaseModel):
    tokens: int  # Tokens used to decide the conditional


class ExecutorFailureFeedback(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    exception: BenchmarkException


ExecutorFeedback = (
    ExecutorOkFeedback
    | ExecutorFailureFeedback
    | ExecutorRunBranch
    | ExecutorSkipBranch
)
