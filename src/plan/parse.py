import json
import re
from typing import Any, Optional, Set, Union, dict, list

import networkx as nx
import regex
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import JsonOutputParser
from plan.domain import PlanStep, PlanType
from plan.exceptions import MisgeneratedPlanException
from pydantic import ValidationError

_PARSER = JsonOutputParser()


def llm_text_to_json(llm_text: str) -> Optional[Union[list[dict], dict]]:
    json_text = llm_text.replace("'", '"').replace("\n'", "")
    try:
        # Try to discard text around the JSON output of interest
        matches = regex.compile(r"\{(?:[^{}]|(?R))*\}").findall(json_text)
        if len(matches) == 1:
            return _PARSER.parse(matches[0])
        return [_PARSER.parse(mt) for mt in matches]
    except Exception as e:
        raise ValueError("Could not parse JSON from text") from e


def parse_json_llm_plan(llm_text: str, tokens: int = 0) -> PlanType:
    try:
        parsed_json = llm_text_to_json(llm_text)
        if parsed_json is None:
            raise MisgeneratedPlanException(
                message="The plan cannot be parsed as a JSON",
                output=llm_text,
                tokens=tokens,
            )
        if isinstance(parsed_json, dict):
            parsed_json = [parsed_json]
        if len(parsed_json) == 0:
            raise OutputParserException(error=f"Could not parse JSON out of {llm_text}")
        plan: PlanType = []
        for json_step in parsed_json:
            if "condition" in json_step:
                # Expecting conditional branch
                step = [PlanStep.model_validate(tool) for tool in json_step["tools"]]
                step[0].evaluate_condition = json_step["condition"]
                step[0].raw_plan_text = json.dumps(json_step)
                if len(json_step) == 0:
                    raise MisgeneratedPlanException(
                        message="Conditional branch should not be empty",
                        output=llm_text,
                        tokens=tokens,
                    )
            else:
                step = PlanStep.model_validate(json_step)
                step.raw_plan_text = json.dumps(json_step)
            plan.append(step)
        return plan
    except OutputParserException as e:
        raise MisgeneratedPlanException(
            message="Could not parse JSON out of text",
            output=llm_text,
            tokens=tokens,
        ) from e
    except ValidationError as e:
        raise MisgeneratedPlanException(
            message=(
                "Output object does not satisfy schema. It must be a "
                f"list of objects that match JSON schema:\n{PlanStep.model_json_schema()}\n"
                "Make sure the object is not nested and \"\" are used, not ''"
            ),
            output=llm_text,
            tokens=tokens,
        ) from e
    except ValueError as e:
        raise MisgeneratedPlanException(
            message="Could not find a json in text",
            output=llm_text,
            tokens=tokens,
        ) from e


def parse_gml_llm_plan(llm_text: str, tokens: int = 0) -> PlanType:
    plan_graph = _gml_to_nx_graph(llm_text, tokens)
    id_to_plan_step: dict[int, PlanStep] = _parse_steps_from_graph(llm_text, plan_graph)
    return _topological_sort_plan_steps(id_to_plan_step, plan_graph)


def _gml_to_nx_graph(llm_text: str, tokens: int) -> nx.Graph:
    llm_text = llm_text.strip()
    glm_text = re.search(r"graph \[.*\]", llm_text)
    if glm_text is None:
        raise MisgeneratedPlanException(
            message="Plan cannot be parsed as a GML",
            output=llm_text,
            tokens=tokens,
        )
    try:
        plan_graph: nx.Graph = nx.parse_gml(llm_text, label=None)
    except nx.NetworkXError as e:
        raise MisgeneratedPlanException(
            message="Plan cannot be parsed as a GML",
            output=llm_text,
            tokens=tokens,
        ) from e
    if not plan_graph.is_directed():
        raise MisgeneratedPlanException(
            message="GLM plan misses the 'directed 1' directive",
            output=llm_text,
            tokens=tokens,
        )
    try:
        _ = nx.find_cycle(plan_graph)
        raise MisgeneratedPlanException(
            message="GLM graph contains cycles",
            output=llm_text,
            tokens=tokens,
        )
    except nx.NetworkXNoCycle:
        return plan_graph


def _parse_steps_from_graph(llm_text: str, plan_graph: nx.Graph) -> dict[int, PlanStep]:
    id_to_plan_step: dict[int, PlanStep] = {}
    for node_id in nx.topological_sort(plan_graph):
        raw_node_text = re.search(rf"node \[ id {node_id} .+? \]", llm_text)
        assert raw_node_text, "A node included in the plan should be found text"
        # pylint: disable=protected-access
        node_metadata: dict[str, Any] = plan_graph._node[node_id]
        args = {}
        for k, v in node_metadata.items():
            if k in ["function", "reason"]:
                continue
            if v == "null":
                args[k] = None
            elif v in ["true", "false"]:
                args[k] = v == "true"
            else:
                args[k] = v
        if len(args) == 0:
            args = None

        id_to_plan_step[node_id] = PlanStep(
            tool_name=node_metadata["function"],
            args=args,
            raw_plan_text=raw_node_text.group(),
            memory=str(node_id),
            reason=node_metadata.get("reason"),
        )
    return id_to_plan_step


def _topological_sort_plan_steps(
    id_to_plan_step: dict[int, PlanStep], plan_graph: nx.Graph
):
    plan: PlanType = []
    visited: Set[int] = set()
    for node_id in nx.topological_sort(plan_graph):
        if node_id in visited:
            continue
        visited.add(node_id)
        plan.append(id_to_plan_step[node_id])
        for _, next_node, edge_metadata in plan_graph.edges(node_id, data=True):
            if "condition" in edge_metadata:
                # Gather all nodes from the conditional branch and put them in a sublist
                branch_first_step = id_to_plan_step[next_node]
                branch_first_step.evaluate_condition = edge_metadata["condition"]
                branch = [branch_first_step]
                visited.add(next_node)
                for _, branch_next_node_id in nx.dfs_edges(plan_graph, next_node):
                    visited.add(branch_next_node_id)
                    branch.append(id_to_plan_step[branch_next_node_id])
                plan.append(branch)
    return plan
