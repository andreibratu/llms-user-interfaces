import json

from overrides import overrides
from pydantic import TypeAdapter

from src.llm import LLMMessage
from src.plan.domain import GeneratedPlanStep, PlanStep
from src.plan.parse import parse_gml_llm_plan, parse_json_llm_plan
from src.strategy.generate import GenerateStrategy


class GenerateCOTOffline(GenerateStrategy):
    def __init__(self, **kwargs) -> None:
        assert "is_online" not in kwargs and "strategy_name" not in kwargs, (
            "Do not override is_online or strategy_name. They are "
            "configured in the constructor of each GenerateStrategy."
        )
        super().__init__(
            strategy_name="cot",
            is_online=False,
            **kwargs,
        )

    @overrides
    def generate(self) -> tuple[str, GeneratedPlanStep]:
        self._build_init_messages()
        self._llm_chat.append(
            LLMMessage(
                role="user",
                content=(
                    "How would you solve this query? Think it step by step and create a "
                    "list of changes you must make to the state of the system."
                ),
            )
        )
        response = self.planner_llm.invoke(
            self._llm_chat,
            **self._llm_hypers,
        )
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
                    "Translate this plan in a JSON array of tools per the examples above"
                    if "json" in self.plan_format
                    else "Translate this plan in a graph using GML notation. Only output the GML format, starting from ["
                ),
            )
        )
        response = self.planner_llm.invoke(
            self._llm_chat,
            **self._llm_hypers,
        )
        self.used_tokens = response.tokens
        if "json" in self.plan_format:
            steps = parse_json_llm_plan(
                response.text,
            )
        else:
            steps = parse_gml_llm_plan(
                response.text,
            )
        return response.text, steps

    @overrides
    def update(self) -> tuple[str, GeneratedPlanStep]:
        return "", []


class GenerateCOTOnline(GenerateStrategy):
    def __init__(self, **kwargs) -> None:
        assert "is_online" not in kwargs and "strategy_name" not in kwargs, (
            "Do not override is_online or strategy_name. They are "
            "configured in the constructor of each GenerateStrategy."
        )
        super().__init__(
            strategy_name="cot",
            is_online=False,
            **kwargs,
        )

    @overrides
    def generate(self) -> tuple[str, GeneratedPlanStep]:
        self._build_init_messages()
        self._llm_chat.append(
            LLMMessage(
                role="user",
                content=(
                    "How would you solve this query? Think it step by step and create a "
                    "list of changes you must make to the state of the system."
                ),
            )
        )
        response = self.planner_llm.invoke(
            self._llm_chat,
            **self._llm_hypers,
        )
        self._llm_chat.append(
            LLMMessage(
                role="assistant",
                content=response.text,
            )
        )
        self._llm_chat.append(
            LLMMessage(
                role="user",
                content=("Translate the first step of the plan in a JSON tool call"),
            )
        )
        response = self.planner_llm.invoke(
            self._llm_chat,
            **self._llm_hypers,
        )
        # Update here in case parsing fails
        self._used_tokens = response.tokens
        steps = parse_json_llm_plan(
            response.text,
        )
        return response.text, steps

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
                        "Output the next step of the JSON execution plan in teh same format provided in the examples"
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
