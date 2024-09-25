"""OpenAI API interface for the LLM model."""

from typing import Any, Literal

from httpx import HTTPStatusError
from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.configuration import APP_CONFIG
from src.domain import Metadata
from src.llm import LLMInterface, LLMMessage, LLMResponse
from src.plan.exceptions import BenchmarkException, ExceptionCode


class OpenAILLM(LLMInterface):
    def __init__(
        self, model: str, finetuning_strategy: Literal["none", "baseline", "tool_bert"]
    ) -> None:
        super().__init__(finetuning_strategy=finetuning_strategy)
        self._client = OpenAI(api_key=APP_CONFIG.openai.api_key)
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    def metadata(self) -> Metadata:
        return {"model": self._model, "type": "openai"}

    @retry(
        retry=retry_if_not_exception_type(BenchmarkException),
        wait=wait_random_exponential(min=1, max=APP_CONFIG.retry_max_seconds),
        stop=stop_after_attempt(5),
    )
    def invoke(
        self, messages: list[LLMMessage], **kwargs: dict[str, Any]
    ) -> LLMResponse:
        try:
            response: ChatCompletion = self._client.chat.completions.create(  # pyright: ignore [reportCallIssue]
                model=self._model,
                messages=[msg.model_dump() for msg in messages],  # pyright: ignore [reportArgumentType]
                **kwargs,  # pyright: ignore [reportArgumentType]
            )
            assistant_text = response.choices[0].message.content
            assert assistant_text is not None, "Assistant text is empty"
            return LLMResponse(
                text=assistant_text,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "original": response.model_dump(),
                },
                tokens=response.usage.total_tokens,  # pyright: ignore [reportOptionalMemberAccess]
            )
        except (BadRequestError, HTTPStatusError) as e:
            # TODO: Maybe make its own error since BenchmarkException is inappropriate here
            raise BenchmarkException(
                code=ExceptionCode.UNEXPECTED,
                message=f"OpenAI API call failed: {str(e)}",
            ) from e
