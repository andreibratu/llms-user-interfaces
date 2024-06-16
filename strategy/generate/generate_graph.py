import itertools
import json
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import nltk
from overrides import overrides

from domain import Metadata
from llm.base import LLMMessage
from plan.domain import PlanStep, PlanType
from plan.exceptions import MisgeneratedPlanException
from plan.parse import parse_json_llm_plan
from prompts import get_system_prompt
from strategy.generate.generate_strategy import GenerateStrategy
from strategy.notification import NewQueryNotification, StrategyNotification
from tool import tools as TOOL_MODULE


class GenerateGraphAbstract(GenerateStrategy):

    def __init__(
        self,
        wire_producers: bool,
        strategy_name: str,
        is_online_strategy: bool,
        plan_deadline_seconds: int = 120,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
            strategy_name=strategy_name,
            is_online_strategy=is_online_strategy,
        )
        self._topology = nx.DiGraph()
        self._plan_deadline = plan_deadline_seconds
        self.wire_producers = wire_producers
        self._curr_node = "START"
        self._build_topology(wire_producers)

    @overrides
    def notify(self, notification: StrategyNotification):
        super().notify(notification)
        if isinstance(notification, NewQueryNotification):
            self._query = notification.query
            self._error = None

    @overrides
    def metadata(self) -> Metadata:
        return {
            **super().metadata(),
            "wire_producers": self.wire_producers,
        }

    @overrides
    def generate(self) -> Tuple[str, PlanType, int]:
        raise NotImplementedError

    @overrides
    def update(self) -> Tuple[str, PlanType, int]:
        raise NotImplementedError

    @overrides
    def _build_system_prompt(self) -> str:
        base_prompt = get_system_prompt(self.plan_format)
        return base_prompt + (
            "\nYou will be guided step by step on what tools you can call. "
            "You begin in START node, and you should go to END node "
            "when you finish a query. "
            "In a function node, call the node to obtain new information."
        )

    def _build_topology(self, wire_producers: bool):
        """
        Build execution graph that assists the LLM plan generation.

        Arguments:
            wire_producers: Add directed edges between nodes (fully-connected)
            skip_functions: Functions that should not be included in the node
        """
        used_tools = list(TOOL_MODULE.TOOL_SCHEMA)
        _producers = [tool for tool in used_tools if tool["role"] == "producer"]
        _consumers = [tool for tool in used_tools if tool["role"] == "consumer"]
        _logic = [
            tool
            for tool in used_tools
            if tool["role"] == "logic" and tool["name"] != "conditional_call"
        ]

        self._topology.add_node(
            "START",
            description=("Go here if you have not finished solving the query yet."),
            type="START",
        )

        self._topology.add_node(
            "END", description=("Go here if the query has been finished"), type="END"
        )

        for tool in _producers:
            t_name = tool["name"]
            self._topology.add_node(
                t_name,
                description=TOOL_MODULE.TOOL_HEADERS[t_name],
                type="producer",
            )
            self._topology.add_edge("START", t_name)

        for tool in _logic:
            t_name = tool["name"]
            self._topology.add_node(
                t_name,
                description=TOOL_MODULE.TOOL_HEADERS[t_name],
                type="logic",
            )

        for tool in _consumers:
            t_name = tool["name"]
            self._topology.add_node(
                t_name,
                description=TOOL_MODULE.TOOL_HEADERS[t_name],
                type="consumer",
            )
            # Finish edge
            self._topology.add_edge(t_name, "END")
            # Return edge
            self._topology.add_edge(t_name, "START")
            # Skip connection in case model does not need additional info
            self._topology.add_edge("START", t_name)

        if wire_producers:
            for p_one, p_two in itertools.product(_producers, _producers):
                if p_one["name"] == p_two["name"]:
                    continue
                self._topology.add_edge(p_one["name"], p_two["name"])

        for producer, logic in itertools.product(_producers, _logic):
            self._topology.add_edge(producer["name"], logic["name"])

        for producer, consumer in itertools.product(_producers, _consumers):
            self._topology.add_edge(producer["name"], consumer["name"])

        # Wiring between logic and consumer layer
        for logic, consumer in itertools.product(_logic, _consumers):
            self._topology.add_edge(logic["name"], consumer["name"])

        # Wiring between logic nodes
        for logic_u, logic_v in itertools.product(_logic, _logic):
            if logic_u["name"] != logic_v["name"]:
                self._topology.add_edge(logic_u["name"], logic_v["name"])

    def _get_next_nodes(self) -> Dict[str, str]:
        """Return a list of nodes and their description."""
        return {
            v: self._topology.nodes[v]["description"]
            + f'\nType: {self._topology.nodes[v]["type"]}'
            for _, v in self._topology.edges(self._curr_node)
        }

    def _build_next_node_user_prompt(self) -> str:
        return (
            f"You are in the {self._curr_node} node. The nodes you "
            f"can currently move to are {self._get_next_nodes()}. "
            "Which node will you move to? Output a single word, the name of node."
        )

    def _build_call_function_user_prompt(self) -> str:
        return (
            f"You are now in function node {self._curr_node}, which "
            f"does the following: {self._get_tool_schema(self._curr_node)}. "
            "Describe next plan step in JSON. "
            f"Use JSON schema:\n{PlanStep.model_json_schema()}"
        )

    @classmethod
    def _parse_node_from_text(cls, text: str, nodes: Iterable) -> Optional[str]:
        # Seperate into words to look for nodes
        for token in nltk.tokenize.RegexpTokenizer(r"\w+").tokenize(text):
            if token in nodes:
                return token
        return None

    @classmethod
    def _get_tool_schema(cls, fn_name: str) -> Dict:
        if fn_name == "START":
            return {"description": "Symbolic node from which you start."}
        for schema in TOOL_MODULE.TOOL_SCHEMA:
            if schema["name"] == fn_name:
                return schema
        raise ValueError(f"Unknown node {fn_name}")

    @classmethod
    def _assistant_node_move_message(cls, prev_node: str, next_node: str) -> LLMMessage:
        return LLMMessage(
            role="assistant", content=f"Moved from {prev_node} to {next_node}"
        )

    @classmethod
    def _assistant_calls_fn_message(cls, fn_call_json: str) -> LLMMessage:
        return LLMMessage(role="assistant", content=f"Called function {fn_call_json}")


class GenerateGraphOnline(GenerateGraphAbstract):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            **kwargs,
            strategy_name="GenerateGraphOnline",
            is_online_strategy=True,
        )
        self._plan: List[PlanStep] = []
        self._generation_start = datetime.now()

    @overrides
    def notify(self, notification: StrategyNotification):
        super().notify(notification)
        if isinstance(notification, NewQueryNotification):
            self._curr_node = "START"
            self._plan = []

    @overrides
    def generate(self) -> Tuple[str, PlanType, int]:
        """Online generation of the plan using the topology."""
        tokens = self._build_init_messages()
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(role="user", content=self._build_next_node_user_prompt()),
            ],
            **self._llm_hypers,
        )
        tokens += response.tokens
        next_node = self._parse_node_from_text(response.text, self._topology.nodes)
        if next_node not in self._get_next_nodes().keys():
            raise MisgeneratedPlanException(
                code=38,
                message=f"No edge exists between {self._curr_node} and {next_node}",
                output=response.text,
                tokens=tokens,
            )
        self._llm_chat.append(
            self._assistant_node_move_message(self._curr_node, next_node)
        )
        self._curr_node = next_node
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(
                    role="user", content=self._build_call_function_user_prompt()
                ),
            ],
            **self._llm_hypers,
        )
        tokens += response.tokens
        step = parse_json_llm_plan(response.text, tokens)
        try:
            self._llm_chat.append(
                self._assistant_calls_fn_message(step[0].model_dump_json())
            )
        except IndexError as e:
            raise e
        return response.text, step, tokens

    @overrides
    def update(self) -> Tuple[str, PlanType, int]:
        """Generate next step."""
        tokens = 0
        seconds_d = (datetime.now() - self._generation_start).total_seconds()
        if seconds_d > self._plan_deadline:
            raise MisgeneratedPlanException(
                code=39,
                message="GenerateGraphOnline: Timeout plan generation",
                output="",
                tokens=0,
            )
        self._llm_chat.append(self._build_transition_message())
        while True:
            response = self.planner_llm.invoke(
                messages=[
                    *self._llm_chat,
                    LLMMessage(
                        role="user", content=self._build_next_node_user_prompt()
                    ),
                ],
                **self._llm_hypers,
            )
            next_node = self._parse_node_from_text(response.text, self._topology)
            tokens += response.tokens
            if next_node not in self._get_next_nodes().keys():
                raise MisgeneratedPlanException(
                    code=38,
                    message=f"No edge exists between {self._curr_node} and {response.text}",
                    output=response.text,
                    tokens=tokens,
                )
            self._llm_chat.append(
                self._assistant_node_move_message(self._curr_node, next_node)
            )
            self._curr_node = next_node
            if self._curr_node != "START":
                # Stop while loop when after we skip past START node
                break
        if self._curr_node == "END":
            return None, [], tokens
        response = self.planner_llm.invoke(
            [
                *self._llm_chat,
                LLMMessage(
                    role="user", content=self._build_call_function_user_prompt()
                ),
            ],
            **self._llm_hypers,
        )
        tokens += response.tokens
        step = parse_json_llm_plan(response.text, tokens)
        self._llm_chat.append(
            LLMMessage(role="assistant", content=step[0].model_dump_json())
        )
        return response.text, step, tokens


class GenerateGraphOffline(GenerateGraphAbstract):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            **kwargs,
            skip_nodes=[],
            strategy_name="GenerateGraphOffline",
            is_online_strategy=False,
        )
        self._memory: List[str] = []
        self._generated_plan: List[PlanStep] = []
        self._curr_node: str = None

    @overrides
    def notify(self, notification: StrategyNotification):
        super().notify(notification)
        if isinstance(notification, NewQueryNotification):
            self._memory = []
            self._generated_plan = []

    @overrides
    def generate(self) -> Tuple[str, PlanType, int]:
        """Generate full plan ahead using the topology."""
        start_t = datetime.now()
        tokens = self._build_init_messages()
        self._curr_node = "START"
        while self._curr_node != "END":
            while True:
                delta_seconds = (datetime.now() - start_t).total_seconds()
                if delta_seconds > 120:
                    raise MisgeneratedPlanException(
                        code=39,
                        message="Timeout on generation GenerateGraphOffline",
                        output=json.dumps(
                            [step.model_dump() for step in self._generated_plan]
                        ),
                        tokens=tokens,
                    )
                # Do while until we end up on node that is not start
                response = self.planner_llm.invoke(
                    [
                        *self._llm_chat,
                        LLMMessage(
                            role="user", content=self._build_next_node_user_prompt()
                        ),
                    ],
                    **self._llm_hypers,
                )
                tokens += response.tokens
                next_node = self._parse_node_from_text(
                    response.text, self._topology.nodes
                )
                self._llm_chat.append(
                    LLMMessage(role="assistant", content=f"Move to node {next_node}")
                )
                if next_node not in self._get_next_nodes().keys():
                    raise MisgeneratedPlanException(
                        code=38,
                        message=f"No edge exists between {self._curr_node} and {next_node}",
                        output=response.text,
                        tokens=tokens,
                    )
                self._curr_node = next_node
                if self._curr_node != "START":
                    break
            if self._curr_node == "END":
                # End condition
                break
            # If in a normal node, we call the function inside
            response = self.planner_llm.invoke(
                [
                    *self._llm_chat,
                    LLMMessage(
                        role="user", content=self._build_call_function_user_prompt()
                    ),
                ],
                **self._llm_hypers,
            )
            tokens += response.tokens
            step = parse_json_llm_plan(response.text, tokens)
            self._llm_chat.append(
                LLMMessage(
                    role="assistant",
                    content=f"Calling function {step[0].model_dump_json()}",
                )
            )
            if step[0].memory:
                self._llm_chat.append(
                    LLMMessage(
                        role="assistant", content=f"Added {step[0].memory} to memory"
                    )
                )
                self._memory.append(step[0].memory)
            self._generated_plan.append(step[0])
        return response.text, self._generated_plan, tokens

    def update(self) -> Tuple[str, PlanType, int]:
        return None, [], 0
