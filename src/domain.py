from typing import Literal, Union

from pydantic import BaseModel

UserId = str

Person = str

JSONType = str

Metadata = dict[str, Union[dict, list, str, int, float]]


PlanFormat = Literal["json", "json+r", "gml", "gml+r", "gml+r+e"]

LLMErrorFeedbackStrategyType = Literal[
    "NO_FEEDBACK",
    "ERROR_TYPE",
    "ERROR_TYPE+STEP",
]


class Coordinates(BaseModel):
    lat: float
    lng: float

    def __hash__(self) -> int:
        return hash((self.lat, self.lng)) + hash(str(type(self)))


class FinetunedLLMConfig(BaseModel):
    finetune_strategy: Literal["none", "tool_bert", "baseline"]
    plan_format: PlanFormat | None
    model: str
