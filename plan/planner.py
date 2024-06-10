import hashlib
import json
from copy import deepcopy
from time import perf_counter
from typing import Generator, List, Optional

from pydantic import ValidationError

from car_state import CarState
from constants import DEFAULT_ORACLE_LLM_HYPERS
from domain import Metadata, PlanFormat
from llm.base import LLMInterface, LLMMessage
from plan.domain import (
    PlanFailureExecutorNotification,
    PlannerOutput,
    PlanRetryExecutorNotification,
    PlanStep,
    PlanSuccessExecutorNotification,
    PlanType,
)
from plan.evaluation import QueryAttempt, QueryEvaluation
from plan.exceptions import MisgeneratedPlanException
from plan.feedback import (
    ExecutorFailureFeedback,
    ExecutorFeedback,
    ExecutorOkFeedback,
    ExecutorRunBranch,
)
from plan.parse import llm_text_to_json, parse_gml_llm_plan, parse_json_llm_plan
from strategy.generate.generate_strategy import GenerateStrategy
from strategy.notification import (
    ExceptionNotification,
    InstructionToExecuteNotification,
    NewQueryNotification,
    OkStrategyNotification,
    PlanFailureStrategyNotification,
    PlanRetryStrategyNotification,
    StrategyNotification,
)
from strategy.retry import RetryStrategy
from tool.tools import get_current_date, weather_tool


class PlannerInterface:

    def __init__(self, llm: LLMInterface) -> None:
        self._evaluation: QueryEvaluation = None
        self._feedback: ExecutorFeedback = None
        self._curr_attempt: QueryAttempt = None
        self._attempts: List[QueryAttempt] = []
        self.llm = llm

    @property
    def identifier(self) -> str:
        raise NotImplementedError

    @property
    def evaluation(self) -> QueryEvaluation:
        assert self._evaluation, "Planner has not finished."
        return self._evaluation

    def _get_feedback(self) -> Optional[ExecutorFeedback]:
        ret_val = self._feedback.model_copy()
        self._feedback = None
        return ret_val

    def post_feedback(self, feedback: ExecutorFeedback) -> None:
        self._feedback = feedback

    def make_plan(self, query: str) -> Generator[PlannerOutput, None, None]:
        raise NotImplementedError

    _ALIGNMENT_SYSTEM_PROMPT = (
        "The state of a car can be described by the following "
        f"schema:\n{CarState.model_json_schema()}\nYou must "
        "predict the impact of user query on the state of the car. "
        "You can guess values and add values to arrays but you "
        "cannot modify keys or the structure of the object. "
        "Use the 'speak' attribute to model interactions between car and driver "
        "and the 'conversations' section to model messages sent by driver to contacts. "
    )

    _ALIGNMENT_USER_PROMPT = (
        "The current state of the car is:\n{initial_car_state}\n"
        "The current time is {current_time}\n"
        "The current weather is {current_weather}\n"
        "The user ask the car assistant the following: {user_query}.\n"
        "Output a JSON object that represents the end state desired "
        "by the user."
    )

    @classmethod
    def _predict_endstate_for_alignment(
        cls, query: str, oracle_llm: LLMInterface
    ) -> Optional[CarState]:
        err, err_count = "", 0
        errors: List[LLMMessage] = []
        # Try multiple times to generate the prediction
        while err_count < 3:
            if err != "":
                errors.append(
                    LLMMessage(
                        role="assistant",
                        content=(
                            f"I tried solving the task but failed with error "
                            f"{err}. I need to avoid this error"
                        ),
                    )
                )
            response = oracle_llm.invoke(
                [
                    LLMMessage(role="system", content=cls._ALIGNMENT_SYSTEM_PROMPT),
                    *errors,
                    LLMMessage(
                        role="user",
                        content=cls._ALIGNMENT_USER_PROMPT.format(
                            user_query=query,
                            initial_car_state=CarState.get_default().model_dump_json(),
                            current_time=get_current_date(),
                            current_weather=weather_tool(),
                        ),
                    ),
                ],
                **DEFAULT_ORACLE_LLM_HYPERS,
            )
            try:
                json_obj = llm_text_to_json(response.text)
                return CarState.model_validate(json_obj)
            except ValidationError as e:
                err_count += 1
                err = str(e)
            except ValueError as e:
                err_count += 1
                err = str(e) + f" {response.text}"
        return None


class SeedPlanner(PlannerInterface):
    """Dummy planner for evaluating seed examples that are human annotated."""

    def __init__(self, llm: LLMInterface, mode: PlanFormat) -> None:
        super().__init__(llm)
        self._mode = mode
        self._query: Optional[str] = None
        self._plan_text: Optional[str] = None

    @property
    def identifier(self) -> str:
        return "SeedPlanner"

    def inject_query_plan(self, query: str, plan_text: str) -> None:
        self._query = query
        self._plan_text = plan_text

    def post_feedback(self, feedback: ExecutorFeedback) -> None:
        self._feedback = feedback

    def make_plan(self, query: str) -> Generator[PlannerOutput, None, None]:
        assert (
            self._query is not None and self._plan_text is not None
        ), "Inject query and plan text before calling make_plan"
        assert self._query == query, "Query should match injected query"
        self._evaluation = None
        predicted_state_alignment = self._predict_endstate_for_alignment(
            query, self.llm
        )
        plan_raw_text = self._plan_text

        if self._mode in ("json", "json+r"):
            plan = parse_json_llm_plan(plan_raw_text)
        elif self._mode in ("gml", "gml+r", "gml+r+e"):
            plan = parse_gml_llm_plan(plan_raw_text)

        curr_attempt = QueryAttempt(
            predicted_state_alignment=predicted_state_alignment,
            car_states=[CarState.get_default()],
            memory_states=[{}],
        )
        curr_attempt.intended_plan = deepcopy(
            [step for step in plan if isinstance(step, PlanStep)]
        )
        curr_attempt.used_tokens = 0
        curr_attempt.raw_llm_text.append(plan_raw_text)

        while len(plan) != 0:
            t1 = perf_counter()
            # Yield plan steps one at a time
            step = plan.pop(0)
            # Send instruction to executor and get feedback
            yield step
            feedback = self._get_feedback()
            assert feedback, "Executor should have sent feedback for current step"
            curr_attempt.time_taken_ms += (perf_counter() - t1) * 1000

            # Executor failed at running the step
            if isinstance(feedback, ExecutorFailureFeedback):
                raise RuntimeError("Should not fail on seed examples")

            # Executor succeeded at running the step
            if isinstance(feedback, ExecutorOkFeedback):
                curr_attempt.executed_plan.append(feedback.step.model_copy())
                curr_attempt.car_states.append(feedback.transition.new_state)
                curr_attempt.tool_output.append(
                    (feedback.step.tool_name, feedback.tool_output)
                )
                curr_attempt.memory_states.append(feedback.transition.new_memory)

            # Executor decided to execute a conditional branch
            if isinstance(feedback, ExecutorRunBranch):
                # Executor wants to pursue branch, add steps to buffer
                curr_attempt.intended_plan.extend(feedback.steps)
                plan = [*feedback.steps, *plan]
                continue

        self._evaluation = QueryEvaluation(
            query=query, attempts=[curr_attempt.model_copy()]
        )
        assert self._evaluation.success, "Seed examples should always succeed"
        yield PlanSuccessExecutorNotification()
        yield
        return "Planner has issued InstructionFinish() already"


class LLMPlanner(PlannerInterface):

    def __init__(
        self,
        retry_strategy: RetryStrategy,
        generation_strategy: GenerateStrategy,
        llm: LLMInterface,
    ) -> None:
        super().__init__(llm)
        self.retry_strategy = retry_strategy
        self.generator_strategy = generation_strategy
        self._feedback: ExecutorFeedback = None

    @property
    def metadata(self) -> Metadata:
        return {
            "generator": self.generator_strategy.metadata(),
            "retry": self.retry_strategy.metadata(),
            "llm": self.llm.metadata(),
        }

    @property
    def identifier(self) -> str:
        hash_f = hashlib.sha1()
        hash_f.update(json.dumps(self.metadata, sort_keys=True).encode("utf-8"))
        return hash_f.hexdigest()

    def post_feedback(self, feedback: ExecutorFeedback) -> None:
        self._feedback = feedback

    @property
    def evaluation(self) -> QueryEvaluation:
        assert self._evaluation, "Planner has not finished."
        return self._evaluation

    def _notify_strategies(self, notification: StrategyNotification) -> None:
        self.generator_strategy.notify(notification)
        self.retry_strategy.notify(notification)

    # pylint: disable=too-many-statements
    def make_plan(self, query: str) -> Generator[PlannerOutput, None, None]:
        self._evaluation = None
        predicted_state_alignment = self._predict_endstate_for_alignment(
            query, self.llm
        )
        self._notify_strategies(
            NewQueryNotification(
                query=query,
                alignment_state=predicted_state_alignment,
            )
        )
        attempts: List[QueryAttempt] = []
        while self.retry_strategy.should_retry() and not any(
            att.error is None for att in attempts
        ):
            plan: PlanType = []
            curr_attempt = QueryAttempt(
                predicted_state_alignment=predicted_state_alignment,
                car_states=[CarState.get_default()],
                memory_states=[{}],
            )

            try:
                # Generate initial plan
                t1 = perf_counter()
                raw_llm_text, plan, tokens = self.generator_strategy.generate()
                curr_attempt.time_taken_ms += (perf_counter() - t1) * 1000
                # Do not copy conditional branches, they will be
                # added as we go to the intended plan
                curr_attempt.intended_plan = deepcopy(
                    [step for step in plan if isinstance(step, PlanStep)]
                )
                curr_attempt.used_tokens += tokens
                curr_attempt.raw_llm_text.append(raw_llm_text)
            except MisgeneratedPlanException as err:
                curr_attempt.time_taken_ms += (perf_counter() - t1) * 1000
                if hasattr(err, "tokens"):
                    curr_attempt.used_tokens += err.tokens
                curr_attempt.error = err.__dict__
                attempts.append(curr_attempt.model_copy())
                self._notify_strategies(ExceptionNotification(exception=err))
                self._notify_strategies(PlanRetryStrategyNotification())
                yield PlanRetryExecutorNotification()
                continue

            while len(plan) != 0:
                t1 = perf_counter()
                # Yield plan steps one at a time
                instruction_step = plan.pop(0)

                if isinstance(instruction_step, PlanStep):
                    # Not conditional branch
                    self._notify_strategies(
                        InstructionToExecuteNotification(instruction=instruction_step)
                    )
                # Send instruction to executor and get feedback
                yield instruction_step
                feedback = self._get_feedback()
                assert feedback, "Executor should have sent feedback for current step"
                curr_attempt.time_taken_ms += (perf_counter() - t1) * 1000

                # Executor failed at running the step
                if isinstance(feedback, ExecutorFailureFeedback):
                    # Cancel current attempt
                    curr_attempt.error = feedback.exception.__dict__
                    attempts.append(curr_attempt)
                    self._notify_strategies(
                        ExceptionNotification(exception=feedback.exception)
                    )
                    self._notify_strategies(PlanRetryStrategyNotification())
                    yield PlanRetryExecutorNotification()
                    break

                # Executor succeeded at running the step
                if isinstance(feedback, ExecutorOkFeedback):
                    curr_attempt.executed_plan.append(feedback.step.model_copy())
                    curr_attempt.car_states.append(feedback.transition.new_state)
                    curr_attempt.tool_output.append(
                        (feedback.step.tool_name, feedback.tool_output)
                    )
                    curr_attempt.memory_states.append(feedback.transition.new_memory)
                    self._notify_strategies(
                        OkStrategyNotification(
                            step=feedback.step,
                            tool_output=feedback.tool_output,
                            transition=feedback.transition,
                        )
                    )

                # Executor decided to execute a conditional branch
                if isinstance(feedback, ExecutorRunBranch):
                    # Executor wants to pursue branch, add steps to buffer
                    curr_attempt.intended_plan.extend(feedback.steps)
                    plan = [*feedback.steps, *plan]
                    continue

                # Allow the strategy to amend steps if online strategy
                try:
                    t1 = perf_counter()
                    raw_llm_text, plan_extension, tokens = (
                        self.generator_strategy.update()
                    )
                    curr_attempt.time_taken_ms += (perf_counter() - t1) * 1000
                    plan.extend(plan_extension)
                    curr_attempt.intended_plan.extend(
                        [step for step in plan_extension if isinstance(step, PlanStep)]
                    )
                    curr_attempt.used_tokens += tokens
                    curr_attempt.raw_llm_text.append(raw_llm_text)
                except MisgeneratedPlanException as err:
                    curr_attempt.error = err.__dict__
                    curr_attempt.used_tokens += err.tokens
                    attempts.append(curr_attempt.model_copy())
                    self._notify_strategies(ExceptionNotification(exception=err))
                    self._notify_strategies(PlanRetryStrategyNotification())
                    yield PlanRetryExecutorNotification()
                    continue

            # Either steps were finished or executor signaled failure
            attempts.append(curr_attempt)

        assert len(attempts) != 0
        # TODO: Change generation to have llm_text upfront
        self._evaluation = QueryEvaluation(query=query, attempts=attempts)
        if not self._evaluation.success:
            self._notify_strategies(PlanFailureStrategyNotification())
        yield (
            PlanSuccessExecutorNotification()
            if self._evaluation.success
            else PlanFailureExecutorNotification()
        )

        # Calling next() after this point will result in failure
        yield
        return "Planner has issued InstructionFinish() already"
