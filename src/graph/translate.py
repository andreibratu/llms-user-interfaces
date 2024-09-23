import json
import re
from collections import defaultdict
from inspect import Signature, getmembers, signature
from typing import dict, list, Optional, tuple
from uuid import uuid4

import networkx as nx
from networkx.classes.reportviews import NodeView

from graph.domain import EdgeType, NodeType
from plan.domain import PlanStep
from tool import tools

_TOOLS = dict(getmembers(tools))
_TOOLS = {fn_name: _TOOLS[fn_name] for fn_name in tools.TOOL_NAMES}
INIT_STATE = ("__INIT__", 0)

_ToolName = str
_MemoryName = str
_ProducersDict = dict[str, tuple[_MemoryName, EdgeType]]


def _fn_returns(name: str) -> bool:
    """Return true if function signature indicates return can be expected."""
    tool_signature = signature(_TOOLS[name])
    return (
        tool_signature.return_annotation is not None
        and tool_signature.return_annotation != Signature.empty
    )


def _requested_memory_dependencies(
    fn_args: dict[str, str], skip_list: list[str]
) -> list[str]:
    dependencies = []
    for arg_name, arg_value in fn_args.items():
        if arg_name in skip_list:
            continue
        mt = re.search(r"\$(\w+)\$", str(arg_value))
        if mt is None:
            continue
        requested_memory = mt[1].lower()
        dependencies.append(requested_memory)
    return dependencies


def _get_producer_node(producers: _ProducersDict, memory: str) -> Optional[str]:
    """Return the node that produced a memory"""
    for producer, (memory_name, _) in producers.items():
        if memory_name.lower() == memory:
            return producer
    return None


# pylint: disable=too-many-branches
def from_plan_to_graph(
    plan: list[PlanStep],
    producers: Optional[dict[tuple, tuple]] = None,
    tool_apparitions: Optional[dict[str, int]] = None,
    depth=0,
) -> tuple[nx.DiGraph, NodeView]:
    head_node = None
    graph = nx.DiGraph()
    if producers is None:
        producers = {INIT_STATE: ("car_address", EdgeType.REAL)}
    if tool_apparitions is None:
        tool_apparitions = defaultdict(int)
    if depth == 0:
        graph.add_node(INIT_STATE, type=NodeType.REAL)
        head_node = graph.nodes[INIT_STATE]
    for step in plan:
        produced_memory = step.memory
        node_fn_args: _ToolName = step.args
        tool_exists = step.tool_name in _TOOLS
        curr_node = (step.tool_name, tool_apparitions[step.tool_name])
        graph.add_node(
            curr_node,
            type=NodeType.REAL if tool_exists else NodeType.HALLUCINATED,
            depth=depth,
        )
        if head_node is None:
            head_node = curr_node
        tool_apparitions[step.tool_name] += 1
        # Add directed edges from producer to current node
        node_dependencies = _requested_memory_dependencies(
            node_fn_args,
            skip_list=(
                ["branch_one", "branch_two"]
                if step.tool_name == "conditional_call"
                else []
            ),
        )
        for dependency_memory in node_dependencies:
            producer = _get_producer_node(producers, dependency_memory)
            if producer:
                memory_type: EdgeType = producers[producer][1]
                graph.add_edge(producer, curr_node, type=memory_type)
            else:
                # Memory does not previously exist
                graph.add_edge((str(uuid4()), 0), curr_node, type=EdgeType.HALLUCINATED)
        if step.tool_name == "conditional_call":
            graph_one, head_one_graph = from_plan_to_graph(
                [
                    PlanStep(**substep)
                    for substep in json.loads(step.args["branch_one"])
                ],
                producers=producers,
                tool_apparitions=tool_apparitions,
                depth=depth + 1,
            )
            graph: nx.Graph = nx.union(graph, graph_one)
            graph.add_edge(curr_node, head_one_graph, type=EdgeType.CONDITIONAL)

            graph_two, head_two_graph = from_plan_to_graph(
                [
                    PlanStep(**substep)
                    for substep in json.loads(step.args["branch_two"])
                ],
                producers=producers,
                tool_apparitions=tool_apparitions,
                depth=depth + 1,
            )
            graph: nx.Graph = nx.union(graph, graph_two)
            graph.add_edge(curr_node, head_two_graph, type=EdgeType.CONDITIONAL)
        # Add node to list of producers
        if step.memory is not None:
            if not tool_exists:
                producers[curr_node] = (
                    produced_memory.lower(),
                    NodeType.HALLUCINATED,
                )
            else:
                if _fn_returns(step.tool_name):
                    producers[curr_node] = (produced_memory.lower(), EdgeType.REAL)
                else:
                    producers[curr_node] = (produced_memory.lower(), EdgeType.ILLEGAL)

    return graph, head_node
