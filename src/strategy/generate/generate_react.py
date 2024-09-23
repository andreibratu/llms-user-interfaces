from datetime import datetime
from typing import Optional, list, tuple

from overrides import overrides

from src.llm.base import LLMMessage
from src.plan.domain import PlanStep
from src.plan.exceptions import MisgeneratedPlanException
from src.plan.parse import parse_json_llm_plan
from src.prompts import get_system_prompt
from src.strategy.generate.generate_strategy import GenerateStrategy
from src.strategy.notification import NewQueryNotification, StrategyNotification


class GenerateReact(GenerateStrategy):
    def __init__(self, timeout_generation: int = 120, **kwargs) -> None:
        super().__init__(
            **kwargs,
            strategy_name="GenerateReact",
            is_online_strategy=True,
        )
        self._timer = datetime.now()
        self._timeout_generation = timeout_generation
        self._generated_plan: list[PlanStep] = []

    @overrides
    def notify(self, notification: StrategyNotification):
        super().notify(notification)
        if isinstance(notification, NewQueryNotification):
            self._generated_plan = []

    @overrides
    def generate(self) -> tuple[list[PlanStep], int]:
        self._timer = datetime.now()
        tokens = self._build_init_messages()
        step, tokens = self._react_step(tokens)
        if step is None:
            return [], tokens
        return [step], tokens

    @overrides
    def update(self) -> tuple[list[PlanStep], int]:
        tokens = 0
        delta_seconds = (datetime.now() - self._timer).total_seconds()
        if delta_seconds > self._timeout_generation:
            # TODO: remove timeout
            raise MisgeneratedPlanException(
                code=39,
                message="Timeout on generation GenerateReact",
                output=[step.model_dump() for step in self._generated_plan],
                tokens=tokens,
            )
        transition_msg = self._build_transition_message()
        # Making this message Observation in accordance with ReAct framework
        transition_msg.content = f"Observation: {transition_msg.content}"
        self._llm_chat.append(transition_msg)
        step, tokens = self._react_step(tokens)
        if step is None:
            return [], tokens
        return [step], tokens

    @classmethod
    def _thought_phase(cls) -> LLMMessage:
        return LLMMessage(
            role="user",
            content=(
                "You are in thought phase. Generate a thought on what "
                "you should do next based on current status."
            ),
        )

    @classmethod
    def _act_phase(cls) -> LLMMessage:
        return LLMMessage(
            role="user",
            content=(
                "You are in act phase. Generate a JSON to call one of "
                "the tools to move towards solving the query. Use "
                f"the JSON schema:\n{PlanStep.model_json_schema()}"
            ),
        )

    @overrides
    def _build_system_prompt(self) -> str:
        prompt = get_system_prompt(self.plan_format)
        prompt += (
            "You will generate the plan sovling the query step by step. "
            "At each step you will generate a Tought where you reason what "
            "must be done, an Action which represents the tool you will call "
            "in JSON format. You will be provided an Observation which "
            "contains the result of the call."
        )
        return prompt

    def _react_step(self, tokens) -> tuple[Optional[PlanStep], int]:
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(
                    role="user",
                    content="Have you finished the query? Respond with YES or NO only.",
                ),
            ]
        )
        tokens += response.tokens
        if "yes" in response.text.lower():
            return None, tokens
        response = self.planner_llm.invoke([*self._llm_chat, self._thought_phase()])
        tokens += response.tokens
        self._llm_chat.append(
            LLMMessage(role="assistant", content="Thought: {response.text}")
        )
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                self._act_phase(),
            ]
        )
        tokens += response.tokens
        step = parse_json_llm_plan(llm_text=response.text, tokens=tokens)[0]
        self._llm_chat.append(
            LLMMessage(
                role="assistant",
                content=f"Act: Calling tool {step.model_dump_json()}",
            )
        )
        return step, tokens
