from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator

from domain import Metadata


class LLMResponse(BaseModel):
    text: Optional[str] = ""
    tokens: int
    metadata: Dict[str, Any]

    @field_validator("text")
    @classmethod
    def text_validator(cls, val):
        return val or ""


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMInterface:

    def invoke(
        self, messages: List[LLMMessage], **kwargs: Dict[str, Any]
    ) -> LLMResponse:
        raise NotImplementedError

    def metadata(self) -> Metadata:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
