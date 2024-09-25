import difflib
import json
import random
import typing
from typing import Literal

from src.car_state import CarState
from src.dataset import read_seed_dataset
from src.domain import LLMErrorFeedbackStrategyType, Metadata, PlanFormat
from src.llm import LLMInterface, LLMMessage
from src.plan.domain import GeneratedPlanStep, PlanStep, Transition
from src.plan.exceptions import BenchmarkException
from src.prompts import get_system_message
from src.strategy import PlannerStrategy
from src.strategy.notification import (
    ExceptionNotification,
    InstructionToExecuteNotification,
    NewQueryNotification,
    OkStrategyNotification,
    StrategyNotification,
)
from src.util import shorten_string_values


class GenerateStrategy(PlannerStrategy):
    _DIFFER = difflib.Differ()

    def __init__(
        self,
        planner_llm: LLMInterface,
        llm_hypers: dict,
        num_demonstrations: int,
        err_feedback_strategy: LLMErrorFeedbackStrategyType,
        is_online: bool,
        plan_format: PlanFormat,
        strategy_name: Literal["blind", "cot", "react"],
    ) -> None:
        self.validate_arguments(
            is_online=is_online,
            plan_format=plan_format,
            error_feedback_strategy=err_feedback_strategy,
        )
        self.is_online_strategy = is_online
        self.plan_format = plan_format
        self.planner_llm = planner_llm
        self._llm_hypers = llm_hypers
        self._error: BenchmarkException | None = None
        self._transitions: list[Transition] = []
        self._llm_chat: list[LLMMessage] = []
        self._step_to_execute: PlanStep | None = None
        self.error_feedback_strategy: LLMErrorFeedbackStrategyType = (
            err_feedback_strategy
        )
        self._query: str | None = None
        self.strategy_name: str = strategy_name
        self.num_demonstrations = num_demonstrations
        self.used_tokens = 0

    @classmethod
    def validate_arguments(
        cls,
        is_online: bool,
        plan_format: PlanFormat,
        error_feedback_strategy: str,
    ):
        if error_feedback_strategy not in typing.get_args(LLMErrorFeedbackStrategyType):
            raise ValueError(
                f"Invalid error feedback strategy: {error_feedback_strategy}"
            )
        if plan_format not in typing.get_args(PlanFormat):
            raise ValueError(f"Invalid plan format: {plan_format}")
        if is_online and "gml" in plan_format:
            raise ValueError("GML planning mode can only work with offline strategies")

    def metadata(self) -> Metadata:
        return {
            "num_demonstrations": self.num_demonstrations,
            "error_feedback_strategy": self.error_feedback_strategy,
            "is_online_strategy": self.is_online_strategy,
            "plan_format": self.plan_format,
            "generation_strategy_name": self.strategy_name,
        }

    def notify(self, notification: StrategyNotification):
        if isinstance(notification, InstructionToExecuteNotification):
            if isinstance(notification, list):
                return
            self._step_to_execute = notification.step
        if isinstance(notification, ExceptionNotification):
            self._error = notification.exception
        if isinstance(notification, OkStrategyNotification):
            self._transitions.append(notification.transition)
        if isinstance(notification, NewQueryNotification):
            self._query = notification.query
            self._transitions = []
            self._llm_chat = []
            self._error = None
            self.used_tokens = 0

    def generate(self) -> tuple[str, GeneratedPlanStep]:
        """Initial generation of the plan.

        Offline strategies will return all their steps at once here.
        Online strategies will return only the first step and wait
        for feedback.
        """
        raise NotImplementedError

    def update(self) -> tuple[str, GeneratedPlanStep]:
        """Amend the initial generation. Used by online strategies."""
        raise NotImplementedError

    def _build_error_explanation(self) -> LLMMessage:
        """Reattempt a query by providing feedback on the error from last attempt."""
        if self.error_feedback_strategy == "NO_FEEDBACK":
            raise ValueError("No feedback strategy selected.")
        explain_error = "There has been an error in your previous attempt."
        if self.error_feedback_strategy in ["ERROR_TYPE", "ERROR_TYPE+STEP"]:
            explain_error += f" Plan has failed with error {self._error}."
            if self.error_feedback_strategy == "ERROR_TYPE+STEP":
                if self._step_to_execute:
                    explain_error += (
                        "The step that failed is "
                        f"{self._step_to_execute.model_dump_json()}. "
                    )
                else:
                    explain_error += (
                        "The error was in plan generation. Be "
                        "sure to respect the format of the structured output. "
                    )
        return LLMMessage(role="user", content=explain_error)

    def _last_run_failed(self) -> bool:
        return self._error is not None

    def _prompt_demonstrations(self) -> LLMMessage:
        """Pick self.num_demonstrations random examples from the seed dataset
        and display them in the initial prompt."""
        seed_dataset = read_seed_dataset(self.plan_format)

        picked_examples = random.choices(
            list(seed_dataset.items()),
            k=self.num_demonstrations,
        )
        output = []
        for query_plan_tuple in picked_examples:
            output.append(
                (f"QUERY: {query_plan_tuple[0]}\n" f"PLAN: {query_plan_tuple[1]}")
            )
        return LLMMessage(
            role="system",
            content="=====\nHere are some plan examples:\n=====\n"
            + "=====\n".join(output),
        )

    def _build_init_messages(self):
        """
        Build the initial messages for the LLM chat. Provides the language agent
        with the system prompt, the problem is must solve, demonstrations and
        the structure it must use to describe the tool calls.
        """
        self._llm_chat = [
            get_system_message(self.plan_format),
            self._prompt_demonstrations(),
        ]
        if self._error is not None and self.error_feedback_strategy != "NO_FEEDBACK":
            self._llm_chat.append(self._build_error_explanation())
        self._llm_chat.append(
            LLMMessage(role="user", content=f"You are solving query {self._query}.")
        )

    def _build_transition_messages(self) -> LLMMessage:
        """Build a message that explains the changes in the car and memory state.

        Used in online strategies to mimic the self-prompt step found in general
        language agent literature.
        """
        if len(self._transitions) == 0:
            return LLMMessage(
                role="assistant",
                content="Previous step has not impacted the state of car or memory.",
            )
        assert (
            len(self._transitions) != 0
        ), "Should only be called after feedback from executor returns"
        json_settings = {"indent": 0, "sort_keys": True}
        last_transition = self._transitions[-1]
        current_car_state, current_memory_state = (
            json.dumps(
                shorten_string_values(last_transition.new_state.model_dump()),
                **json_settings,
                ensure_ascii=False,
            ).splitlines(),
            json.dumps(
                shorten_string_values(last_transition.new_memory),
                **json_settings,
                ensure_ascii=False,
            ).splitlines(),
        )
        if len(self._transitions) >= 2:
            old_car_state, old_memory_state = (
                json.dumps(
                    shorten_string_values(self._transitions[-2].new_state.model_dump()),
                    **json_settings,
                    ensure_ascii=False,
                ).splitlines(),
                json.dumps(
                    shorten_string_values(self._transitions[-2].new_memory),
                    **json_settings,
                    ensure_ascii=False,
                ).splitlines(),
            )
        else:
            old_car_state, old_memory_state = (
                json.dumps(
                    shorten_string_values(CarState.get_default().model_dump()),
                    **json_settings,
                    ensure_ascii=False,
                ).splitlines(),
                json.dumps({}, **json_settings, ensure_ascii=False).splitlines(),
            )
        diff_car_state = "\n".join(
            [
                diff
                for diff in self._DIFFER.compare(old_car_state, current_car_state)
                if not diff.startswith(" ")
            ]
        )
        diff_memory_state = "\n".join(
            [
                diff.replace("\\", "")
                for diff in self._DIFFER.compare(old_memory_state, current_memory_state)
                if not diff.startswith(" ")
            ]
        )
        return LLMMessage(
            role="assistant",
            content=(
                "The previous function call has resulted in these changes.\n"
                f"Car state: {diff_car_state}\nMemory state:\n{diff_memory_state}"
            ),
        )
