import re
from datetime import datetime
from typing import List, Tuple

from overrides import overrides

from llm.base import LLMMessage
from plan.domain import PlanStep, PlanType
from plan.exceptions import MisgeneratedPlanException
from plan.parse import parse_json_llm_plan
from strategy.generate.generate_strategy import GenerateStrategy


class GenerateBabyAGI(GenerateStrategy):

    def __init__(
        self,
        timeout_deadline: int = 180,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
            strategy_name="GenerateBabyAGI",
            is_online_strategy=True,
        )
        self._tasks: List[Tuple[int, str]] = [
            (
                0,
                "Create a list of tasks for solving the user query. Return "
                "task items as a numbered list. Higher items in the "
                "list will be solved first.",
            )
        ]
        self._timer = datetime.now()
        self._timeout_deadline = timeout_deadline

    @overrides
    def generate(self) -> Tuple[str, PlanType, int]:
        self._timer = datetime.now()
        tokens = self._build_init_messages()
        _, step, nt = self._handle_top_task()
        tokens += nt
        if step is not None:
            # Generated PlanStep when was expecting to recurse
            raise MisgeneratedPlanException(
                code=43,
                message="Expected to create subtasks instead of calling a tool.",
                output="",
                tokens=tokens,
            )
        while step is None:
            self._assert_before_deadline(tokens)
            llm_text, step, nt = self._handle_top_task()
            tokens += nt
        return llm_text, step, tokens

    @overrides
    def update(self) -> Tuple[str, PlanType, int]:
        self._llm_chat.append(self._build_transition_message())
        tokens = self._reprioritize_tasks()
        step = None
        while step is None:
            self._assert_before_deadline(tokens)
            llm_text, step, nt = self._handle_top_task()
            tokens += nt
        return llm_text, step, tokens

    def _task_list_to_str(self) -> str:
        return "\n".join(
            f"{task_id}. {task_text}" for task_id, task_text in self._tasks
        )

    def _assert_before_deadline(self, tokens_expanded: int):
        if (datetime.now() - self._timer).total_seconds() > self._timeout_deadline:
            raise MisgeneratedPlanException(
                code=39,
                message="GenerateBabyAGI ran out of time",
                output="",
                tokens=tokens_expanded,
            )

    _REPRIORITIZE_TEXT = (
        "You must reprioritise the list of tasks that you want to "
        "address in order to solve the user query. Make sure to not eliminate "
        "tasks out of the list, only reorder them. Output the list of tasks as "
        "a numbered list; higher items in the list will be solved first. "
        "The current list of tasks is:\n"
    )

    _NUMBERED_LIST_ITEM_RE = r"(\d+)\.\s+(.+)"

    @overrides
    def _build_system_prompt(self) -> str:
        base_sys_prompt = super()._build_system_prompt()
        return base_sys_prompt + (
            "You will approach the query by mentaining a list of tasks. "
            "Each task can be subdivided into further natural language subtasks, or "
            "result in a JSON function call as mentioned above. You are doen with the query "
            "when all tasks are finished so make sure to be as step by step as possible, "
            "including communicating the result to the user."
        )

    @overrides
    def _build_transition_message(self) -> LLMMessage:
        # Corresponds to context_agent in the original work
        message = super()._build_transition_message()
        message.content += f"\nThe current task list is {self._task_list_to_str()}"
        return message

    def _reprioritize_tasks(self) -> int:
        # Corresponds to prioritization_agent in the original work
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(
                    role="user",
                    content=self._REPRIORITIZE_TEXT + self._task_list_to_str(),
                ),
            ],
            **self._llm_hypers,
        )
        self._tasks = []
        for match in re.findall(self._NUMBERED_LIST_ITEM_RE, response.text):
            task_id, task_text = match
            self._tasks.append((task_id, task_text))
        self._tasks.sort()
        return response.tokens

    def _handle_top_task(self) -> Tuple[str, PlanType, int]:
        # Corresponds to execution_agent in the original work
        tokens = 0
        if len(self._tasks) == 0:
            return "", [], 0
        top_task = self._tasks.pop(0)
        # Decrement ids of the task
        c_id, c_text = top_task[0], top_task[1]
        for idx, task in enumerate(self._tasks):
            n_id = task[0]
            self._tasks[idx] = (c_id, task[1])
            c_id = n_id

        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(
                    role="user",
                    content=(
                        f"You are solving current task: {c_text}. "
                        "Either break down this task into subtasks. Or "
                        "call a tool that solves this tasks. If you "
                        "want to break it down in tasks follow the format: "
                        "Recurse: Numbered list of subtasks. If you want to "
                        f"call a tool, use format {PlanStep.model_json_schema()}"
                    ),
                ),
            ]
        )
        tokens += response.tokens
        if matches := re.findall(self._NUMBERED_LIST_ITEM_RE, response.text):
            for match in matches:
                if len(self._tasks) > 0:
                    next_id = self._tasks[-1][0]
                else:
                    next_id = 1
                task_text = match[1]
                # Add each new task at end of the list
                self._tasks.append((next_id, task_text))
            self._llm_chat.append(
                LLMMessage(
                    role="assistant",
                    content=f"Added tasks to list. Current list: {self._task_list_to_str()}",
                )
            )
            return None, tokens
        # Assuming this is a tool call
        next_step = parse_json_llm_plan(llm_text=response.text, tokens=tokens)[0]
        self._llm_chat.append(
            LLMMessage(
                role="assistant",
                content=next_step.model_dump_json(),
            )
        )
        return response.text, next_step, tokens
