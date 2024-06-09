from typing import Tuple

from overrides import overrides

from llm.base import LLMMessage
from plan.domain import PlanType
from plan.parse import parse_json_llm_plan
from strategy.generate.generate_strategy import GenerateStrategy


class GenerateCOT(GenerateStrategy):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            strategy_name="GenerateCOT",
            is_online_strategy=False,
            **kwargs,
        )

    @overrides
    def generate(self) -> Tuple[str, PlanType, int]:
        tokens = self._build_init_messages()
        self._llm_chat.append(
            LLMMessage(
                role="user",
                content=(
                    "How would you solve this query? Describe the plan "
                    "step by step in natural language. "
                ),
            )
        )
        response = self.planner_llm.invoke(
            self._llm_chat,
            **self._llm_hypers,
        )
        tokens += response.tokens
        self._llm_chat.append(
            LLMMessage(
                role="assistant",
                content=response.text,
            )
        )
        self._llm_chat.append(
            LLMMessage(
                role="user",
                content=(
                    "Translate this plan in a JSON array of tools per examples"
                    if "json" in self.plan_format
                    else "Translate this plan in a graph using GML notation"
                ),
            )
        )
        response = self.planner_llm.invoke(
            self._llm_chat,
            **self._llm_hypers,
        )
        tokens += response.tokens
        steps = parse_json_llm_plan(
            response.text,
            response.tokens,
        )
        return response.text, steps, tokens

    @overrides
    def update(self) -> Tuple[str, PlanType, int]:
        return None, [], 0
