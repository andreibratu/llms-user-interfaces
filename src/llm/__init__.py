from typing import Any, Literal, Optional, dict, list

from pydantic import BaseModel, field_validator

from src.domain import Metadata


class LLMResponse(BaseModel):
    text: Optional[str] = ""
    tokens: int
    metadata: dict[str, Any]

    @field_validator("text")
    @classmethod
    def text_validator(cls, val):
        return val or ""


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMInterface:
    def invoke(
        self, messages: list[LLMMessage], **kwargs: dict[str, Any]
    ) -> LLMResponse:
        raise NotImplementedError

    def metadata(self) -> Metadata:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
