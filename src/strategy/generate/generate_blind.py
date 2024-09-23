from typing import tuple

from overrides import overrides

from src.llm.base import LLMMessage
from src.plan.domain import PlanStep, PlanType
from src.plan.parse import parse_gml_llm_plan, parse_json_llm_plan
from src.strategy.generate.generate_strategy import GenerateStrategy


class GenerateBlindAbstract(GenerateStrategy):
    def __init__(
        self,
        is_online_strategy: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
            is_online_strategy=is_online_strategy,
        )

    @overrides
    def generate(self) -> tuple[str, PlanType, int]:
        raise NotImplementedError

    @overrides
    def update(self) -> tuple[str, PlanType, int]:
        raise NotImplementedError


class GenerateBlindOffline(GenerateBlindAbstract):
    """Generates full plan given query. Regenerates entire plan given negative feedback."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            **kwargs,
            is_online_strategy=False,
            strategy_name="GenerateBlindOffline",
        )

    @overrides
    def generate(self) -> tuple[str, PlanType, int]:
        # First generation done in an attempt
        init_tokens = self._build_init_messages()
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                (
                    LLMMessage(
                        role="user",
                        content="Generate a plan as dependency graph in GML format",
                    )
                    if self.plan_format
                    else LLMMessage(
                        role="user", content="Generate a plan as JSON array of objects"
                    )
                ),
            ],
            **self._llm_hypers,
        )
        if self.plan_format == "gml":
            return (
                response.text,
                parse_gml_llm_plan(response.text, response.tokens),
                response.tokens + init_tokens,
            )
        return (
            response.text,
            parse_json_llm_plan(response.text, response.tokens),
            response.tokens + init_tokens,
        )

    @overrides
    def update(self) -> tuple[str, PlanType, int]:
        return None, [], 0


class GenerateBlindOnline(GenerateBlindAbstract):
    _FIRST_STEP_PROMPT = (
        "Generate first step of the plan as JSON"
        f"object. Use JSON schema:\n{PlanStep.model_json_schema()}\n"
    )

    def __init__(self, is_online_strategy: bool, **kwargs) -> None:
        super().__init__(
            is_online_strategy=is_online_strategy,
            strategy_name="GenerateBlindOnline",
            **kwargs,
        )

    @overrides
    def generate(self) -> tuple[str, PlanType, int]:
        tokens = self._build_init_messages()
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(
                    role="user",
                    content=self._FIRST_STEP_PROMPT,
                ),
            ]
        )
        tokens += response.tokens
        first_step: PlanType = parse_json_llm_plan(response.text, tokens=tokens)[0]
        self._llm_chat.append(
            LLMMessage(
                role="assistant",
                content=f"{first_step[0].model_dump_json()}",
            )
        )
        return response.text, first_step, tokens

    @overrides
    def update(self) -> tuple[str, PlanType, int]:
        tokens = 0
        self._llm_chat.extend(
            [
                self._build_transition_message(),
                LLMMessage(
                    role="user",
                    content=(
                        "Have you finished the plan to answer the query? "
                        "Output only 'YES' or 'NO'"
                    ),
                ),
            ]
        )
        response = self.planner_llm.invoke(
            messages=self._llm_chat,
            **self._llm_hypers,
        )
        tokens += response.tokens
        if "yes" in response.text.lower():
            return [], tokens
        self._llm_chat.extend(
            [
                LLMMessage(
                    role="user",
                    content=(
                        "Output the next step of the plan as JSON. "
                        f"Use the JSON schema:\n{PlanStep.model_json_schema()}"
                    ),
                )
            ]
        )
        response = self.planner_llm.invoke(self._llm_chat, **self._llm_hypers)
        tokens += response.tokens
        next_step = parse_json_llm_plan(response.text, tokens)[0]
        return response.text, next_step, tokens
