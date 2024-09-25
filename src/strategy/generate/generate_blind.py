import json

from overrides import overrides
from pydantic import TypeAdapter

from src.llm import LLMMessage
from src.plan.domain import GeneratedPlanStep, PlanStep
from src.plan.parse import parse_gml_llm_plan, parse_json_llm_plan
from src.strategy.generate import GenerateStrategy


class GenerateBlindOffline(GenerateStrategy):
    """Generates full plan given query with no additional prompting."""

    def __init__(self, **kwargs) -> None:
        assert "is_online" not in kwargs and "strategy_name" not in kwargs, (
            "Do not override is_online or strategy_name. They are "
            "configured in the constructor of each GenerateStrategy."
        )
        super().__init__(
            **kwargs,
            is_online=False,
            strategy_name="blind",
        )

    @overrides
    def generate(self) -> tuple[str, GeneratedPlanStep]:
        # First generation done in an attempt
        self._build_init_messages()
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                (
                    LLMMessage(
                        role="user",
                        content="Generate a plan as dependency graph in GML format",
                    )
                    if "gml" in self.plan_format
                    else LLMMessage(
                        role="user", content="Generate a plan as JSON array of objects"
                    )
                ),
            ],
            **self._llm_hypers,
        )
        if "gml" in self.plan_format:
            return (
                response.text,
                parse_gml_llm_plan(response.text),
            )
        # Update here in case parsing fails
        self.used_tokens = response.tokens
        return (
            response.text,
            parse_json_llm_plan(response.text),
        )

    @overrides
    def update(self) -> tuple[str, GeneratedPlanStep]:
        return "", []


class GenerateBlindOnline(GenerateStrategy):
    _FIRST_STEP_PROMPT = (
        "Generate first step of the plan as JSON"
        f"object. Use JSON schema:\n{PlanStep.model_json_schema()}\n"
    )

    def __init__(self, **kwargs) -> None:
        assert "is_online" not in kwargs and "strategy_name" not in kwargs, (
            "Do not override is_online or strategy_name. They are "
            "configured in the constructor of each GenerateStrategy."
        )
        super().__init__(
            is_online=True,
            strategy_name="blind",
            **kwargs,
        )

    @overrides
    def generate(self) -> tuple[str, GeneratedPlanStep]:
        self._build_init_messages()
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(
                    role="user",
                    content=self._FIRST_STEP_PROMPT,
                ),
            ]
        )
        # Update here in case parsing fails
        self._used_tokens = response.tokens
        plan: GeneratedPlanStep = parse_json_llm_plan(response.text)
        self._llm_chat.append(
            LLMMessage(
                role="assistant",
                content=json.dumps(
                    TypeAdapter(GeneratedPlanStep).dump_python(plan),
                    ensure_ascii=False,
                ),
            )
        )
        return response.text, plan

    @overrides
    def update(self) -> tuple[str, GeneratedPlanStep]:
        self._llm_chat.extend(
            [
                self._build_transition_messages(),
                LLMMessage(
                    role="user",
                    content=(
                        "Have you finished the plan to answer the query? "
                        "Output only 'yes' or 'no'"
                    ),
                ),
            ]
        )
        response = self.planner_llm.invoke(
            messages=self._llm_chat,
            **self._llm_hypers,
        )
        if "yes" in response.text.lower():
            self._used_tokens = response.tokens
            return "", []
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
        next_step = parse_json_llm_plan(response.text)
        self._llm_chat.append(
            LLMMessage(
                role="assistant",
                content=json.dumps(
                    TypeAdapter(GeneratedPlanStep).dump_python(next_step),
                    ensure_ascii=False,
                ),
            )
        )
        return response.text, next_step
