from typing import Any, Dict, List

from httpx import HTTPStatusError
from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletion
from overrides import override
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from configuration import APP_CONFIG
from domain import Metadata
from llm.base import LLMInterface, LLMMessage, LLMResponse
from plan.exceptions import ToolException


# TODO: logit bias for increased plan likelihood
class OpenAILLM(LLMInterface):

    def __init__(self, model: str) -> None:
        self._client = OpenAI(api_key=APP_CONFIG.openai.api_key)
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    @override
    def metadata(self) -> Metadata:
        return {"model": self._model, "type": "openai"}

    @retry(
        retry=retry_if_not_exception_type(ToolException),
        wait=wait_random_exponential(min=1, max=APP_CONFIG.retry_max_seconds),
        stop=stop_after_attempt(5),
    )
    def invoke(
        self, messages: List[LLMMessage], **kwargs: Dict[str, Any]
    ) -> LLMResponse:
        response = None
        try:
            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=[msg.model_dump() for msg in messages],
                **kwargs
            )
        except (BadRequestError, HTTPStatusError) as e:
            # TODO: Maybe make its own error since ToolException is innapropiate here
            raise ToolException(
                tool_name="openai_invoke",
                code=35,
                taxonomy="openai_throws",
                message=str(e),
            ) from e
        if not response or len(response.choices) == 0:
            return None
        assistant_text = response.choices[0].message.content
        return LLMResponse(
            text=assistant_text,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "original": response.model_dump(),
            },
            tokens=response.usage.total_tokens,
        )