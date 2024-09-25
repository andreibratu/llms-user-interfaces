from datetime import datetime

from overrides import overrides

from src.llm import LLMMessage
from src.plan.domain import GeneratedPlanStep, PlanStep
from src.plan.exceptions import BenchmarkException, ExceptionCode
from src.plan.parse import parse_json_llm_plan
from src.strategy.generate import GenerateStrategy
from src.strategy.notification import NewQueryNotification, StrategyNotification


class GenerateReactOnline(GenerateStrategy):
    def __init__(self, timeout_generation: int = 240, **kwargs) -> None:
        assert "is_online" not in kwargs and "strategy_name" not in kwargs, (
            "Do not override is_online or strategy_name. They are "
            "configured in the constructor of each GenerateStrategy."
        )
        super().__init__(
            **kwargs,
            strategy_name="react",
            is_online=True,
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
    def generate(self) -> tuple[str, GeneratedPlanStep]:
        self._timer = datetime.now()
        self._build_init_messages()
        text, step = self._react_step()
        if step is None:
            return "", []
        return text, [step]

    @overrides
    def update(self) -> tuple[str, GeneratedPlanStep]:
        delta_seconds = (datetime.now() - self._timer).total_seconds()
        if delta_seconds > self._timeout_generation:
            # TODO: remove timeout
            raise BenchmarkException(
                code=ExceptionCode.UNEXPECTED,
                message="Timeout on generation GenerateReact",
            )
        transition_msg = self._build_transition_messages()
        # Observation step in accordance with ReACT algorithm
        transition_msg.content = f"Observation: {transition_msg.content}"
        self._llm_chat.append(transition_msg)
        text, step = self._react_step()
        if step is None:
            return "", []
        return text, [step]

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
    def _build_init_messages(self):
        super()._build_init_messages()
        self._llm_chat.insert(
            # Insert message before the one containing the query
            1,
            LLMMessage(
                role="assistant",
                content=(
                    "You will generate the plan solving the query step by step. "
                    "At each step you will generate a Thought where you reason what "
                    "must be done, an Action which represents the tool you will call "
                    "in JSON format. You will be provided an Observation which "
                    "contains the result of the call."
                ),
            ),
        )

    def _react_step(self) -> tuple[str, PlanStep | None]:
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(
                    role="user",
                    content="Have you finished the query? Respond with 'yes' or 'no' only.",
                ),
            ]
        )
        if "yes" in response.text.lower():
            return "", None
        response = self.planner_llm.invoke([*self._llm_chat, self._thought_phase()])
        self._llm_chat.append(
            LLMMessage(role="assistant", content="Thought: {response.text}")
        )
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                self._act_phase(),
            ]
        )
        self.used_tokens = response.tokens
        step = parse_json_llm_plan(llm_text=response.text)
        assert isinstance(step, PlanStep)
        self._llm_chat.append(
            LLMMessage(
                role="assistant",
                content=f"Act: Calling tool {step.model_dump_json()}",
            )
        )
        return response.text, step
