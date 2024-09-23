from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict

from plan.domain import PlanStep, Transition
from plan.exceptions import BenchmarkException


class ExecutorOkFeedback(BaseModel):
    step: PlanStep
    tool_output: Union[List, Dict, str, None]
    transition: Transition


class ExecutorRunBranch(BaseModel):
    tokens: int
    steps: List[PlanStep]


class ExecutorSkipBranch(BaseModel):
    tokens: int


class ExecutorFailureFeedback(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    exception: BenchmarkException


ExecutorFeedback = Union[
    ExecutorOkFeedback, ExecutorFailureFeedback, ExecutorRunBranch, ExecutorSkipBranch
]
