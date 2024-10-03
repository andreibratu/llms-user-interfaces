"""Abstraction over how different LLM providers"""

from typing import Any, Literal

from pydantic import BaseModel, field_validator

from src.domain import Metadata, PlanFormat


class LLMResponse(BaseModel):
    text: str = ""
    tokens: int  # Total tokens used until this point
    metadata: dict[str, Any]

    @field_validator("text")
    @classmethod
    def text_validator(cls, val):
        return val or ""


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMInterface:
    finetuning_strategy: Literal["none", "baseline", "tool_bert"]

    def __init__(
        self,
        finetune_strategy: Literal["none", "baseline", "tool_bert"],
        finetune_format: PlanFormat | None,
    ) -> None:
        self.finetuning_strategy = finetune_strategy
        self.finetune_format = finetune_format

    def invoke(
        self,
        messages: list[LLMMessage],
        **kwargs: int | float | str,
    ) -> LLMResponse:
        raise NotImplementedError

    def metadata(self) -> Metadata:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
