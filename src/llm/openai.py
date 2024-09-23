from typing import Any

from httpx import HTTPStatusError
from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletion
from plan.exceptions import BenchmarkException, ExceptionCode
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import src.session as SESSION
from src.domain import Metadata
from src.llm import LLMInterface, LLMMessage, LLMResponse


class OpenAILLM(LLMInterface):
    def __init__(self, model: str) -> None:
        self._client = OpenAI(api_key=SESSION.APP_CONFIG.openai.api_key)
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    def metadata(self) -> Metadata:
        return {"model": self._model, "type": "openai"}

    @retry(
        retry=retry_if_not_exception_type(BenchmarkException),
        wait=wait_random_exponential(min=1, max=SESSION.APP_CONFIG.retry_max_seconds),
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
