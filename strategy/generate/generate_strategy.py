import difflib
import json
import random
from typing import Dict, List, Literal, Optional, Tuple

from car_state import CarState
from dataset.read_seed import read_seed_dataset
from domain import Metadata, PlanFormat
from llm.base import LLMInterface, LLMMessage
from plan.domain import PlanStep, PlanType, Transition
from plan.exceptions import ExecutionException
from prompts import get_system_prompt
from strategy.base import PlannerStrategy
from strategy.notification import (
    ExceptionNotification,
    InstructionToExecuteNotification,
    NewQueryNotification,
    OkStrategyNotification,
    StrategyNotification,
)
from util import shorten_string_values

LLMErrorFeedbackStrategyType = Literal[
    "NO_FEEDBACK",
    "ERROR_EXISTS",
    "ERROR_TYPE",
    "ERROR_TYPE+STEP",
]


class GenerateStrategy(PlannerStrategy):

    _DIFFER = difflib.Differ()

    def __init__(
        self,
        planner_llm: LLMInterface,
        llm_hypers: Dict,
        num_demonstrations: int,
        err_feedback_strategy: LLMErrorFeedbackStrategyType,
        use_alignment_prediction: bool,
        is_online_strategy: bool,
        plan_format: PlanFormat,
        strategy_name: str,
    ) -> None:
        super().__init__()
        if is_online_strategy and plan_format == "gml":
            raise ValueError("GML planning mode can only work with offline strategies")
        self.is_online_strategy = is_online_strategy
        self.plan_format = plan_format
        self.planner_llm = planner_llm
        self._llm_hypers = llm_hypers
        self._error: ExecutionException = None
        self._transitions: List[Transition] = []
        self._llm_chat: List[LLMMessage] = []
        self._step_to_execute: PlanStep = []
        self.error_feedback_strategy = err_feedback_strategy
        self._alignment_end_state_prediction: Optional[CarState] = None
        self._query: str = None
        self.strategy_name: str = strategy_name
        self.use_alignment_prediction = use_alignment_prediction
        self.num_demonstrations = num_demonstrations

    def metadata(self) -> Metadata:
        return {
            "num_demonstrations": self.num_demonstrations,
            "use_alignment_prediction": self.use_alignment_prediction,
            "err_feedback_strategy": self.error_feedback_strategy,
            "planner_llm": self.planner_llm.metadata(),
            "is_online_strategy": self.is_online_strategy,
            "plan_output_mode": self.plan_format,
            "strategy_name": self.strategy_name,
        }

    def notify(self, notification: StrategyNotification):
        if isinstance(notification, InstructionToExecuteNotification):
            if isinstance(notification, list):
                return
            self._step_to_execute = notification
        if isinstance(notification, ExceptionNotification):
            self._error = notification.exception
        if isinstance(notification, OkStrategyNotification):
            self._transitions.append(notification.transition)
        if isinstance(notification, NewQueryNotification):
            self._query = notification.query
            self._alignment_end_state_prediction = notification.alignment_state
            self._transitions = []
            self._llm_chat = []
            self._error = None

    def generate(self) -> Tuple[str, PlanType, int]:
        """Initial generation of the plan.

        Offline strategies will return all their steps at once here.
        Online strategies will return only the first step and await
        for feedback.
        """
        raise NotImplementedError

    def update(self) -> Tuple[str, PlanType, int]:
        """Amend the initial generation."""
        raise NotImplementedError

    def _build_error_explain_prompt(self) -> str:
        if self.error_feedback_strategy == "NO_FEEDBACK":
            return ""
        # Case 'ERROR_EXISTS'
        explain_error = "There has been an error in the plan."
        if self.error_feedback_strategy in ["ERROR_TYPE", "ERROR_TYPE+STEP"]:
            explain_error += f" Plan has failed with error {self._error}."
            if self.error_feedback_strategy == "ERROR_TYPE+STEP":
                if self._step_to_execute:
                    # failed_step might be null due to MisgeneratedPlanException
                    explain_error += (
                        "The step that failed is "
                        f"{self._step_to_execute.model_dump_json()}. "
                    )
                else:
                    explain_error += (
                        "The error was in plan generation. Be "
                        "sure to respect the schema. "
                    )
        explain_error += "Can you rephrase? Write the full plan again as JSON list."
        return explain_error

    def _error_last_run(self) -> bool:
        return self._error is not None

    def _prompt_n_examples(self):
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
        return "=====\nHere are some plan examples:\n=====\n" + "=====\n".join(output)

    def _build_init_messages(self) -> int:
        tokens = 0
        self._llm_chat = [
            LLMMessage(role="system", content=get_system_prompt(self.plan_format)),
            LLMMessage(
                role="system",
                content=f"Here are some examples:\n{self._prompt_n_examples()}",
            ),
        ]
        if self._error is not None:
            self._llm_chat.append(
                LLMMessage(
                    role="user",
                    content=self._build_error_explain_prompt(),
                )
            )
        self._llm_chat.append(
            LLMMessage(role="user", content=f"You are solving query {self._query}.")
        )
        if self.use_alignment_prediction and self._alignment_end_state_prediction:
            deltas = "".join(
                line
                for line in difflib.ndiff(
                    CarState.get_default().model_dump_json(indent=4).splitlines(True),
                    self._alignment_end_state_prediction.model_dump_json(
                        indent=4
                    ).splitlines(True),
                )
                # Do not include lines that are same
                if not line.startswith(" ")
            )
            self._llm_chat.append(
                LLMMessage(
                    role="assistant",
                    content=(
                        "I believe the following changes are required to reach "
                        f"the desired end state:\n{deltas}\nI will create a "
                        "plan to make it so"
                    ),
                )
            )
        return tokens

    def _build_transition_message(self) -> LLMMessage:
        if len(self._transitions) == 0:
            return LLMMessage(
                role="assistant",
                content="Previous step has not impacted the state of car or memory.",
            )
        assert (
            len(self._transitions) != 0
        ), "Should only be called after feedback from executor returns"
        json_settings = {"indent": 0, "sort_keys": True}
        last_t = self._transitions[-1]
        current_car_state, current_memory_state = (
            json.dumps(
                shorten_string_values(last_t.new_state.model_dump()), **json_settings
            ).splitlines(),
            json.dumps(
                shorten_string_values(last_t.new_memory), **json_settings
            ).splitlines(),
        )
        if len(self._transitions) >= 2:
            old_car_state, old_memory_state = (
                json.dumps(
                    shorten_string_values(self._transitions[-2].new_state.model_dump()),
                    **json_settings,
                ).splitlines(),
                json.dumps(
                    shorten_string_values(self._transitions[-2].new_memory),
                    **json_settings,
                ).splitlines(),
            )
        else:
            old_car_state, old_memory_state = (
                json.dumps(
                    shorten_string_values(CarState.get_default().model_dump()),
                    **json_settings,
                ).splitlines(),
                json.dumps({}, **json_settings).splitlines(),
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
