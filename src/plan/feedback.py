from typing import dict, list, Union

from pydantic import BaseModel, ConfigDict

from plan.domain import PlanStep, Transition
from plan.exceptions import BenchmarkException


class ExecutorOkFeedback(BaseModel):
    step: PlanStep
    tool_output: Union[list, dict, str, None]
    transition: Transition


class ExecutorRunBranch(BaseModel):
    tokens: int
    steps: list[PlanStep]


class ExecutorSkipBranch(BaseModel):
    tokens: int


class ExecutorFailureFeedback(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    exception: BenchmarkException


ExecutorFeedback = Union[
    ExecutorOkFeedback, ExecutorFailureFeedback, ExecutorRunBranch, ExecutorSkipBranch
]
