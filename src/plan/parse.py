"""Parse LLM output into an execution plan."""

import json
import re
from typing import Any

import networkx as nx
import regex
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import ValidationError

from src.plan.domain import GeneratedPlanStep, PlanStep
from src.plan.exceptions import MisgeneratedPlanException

_PARSER = JsonOutputParser()


def llm_text_to_json(llm_text: str) -> list[dict] | dict | None:
    """Extracted structured JSON output from LLM generation."""
    json_text = llm_text.replace("'", '"').replace("\n'", "")
    json_text = json_text.replace("\\", "")
    try:
        # Try to discard text around the JSON output of interest, extracting inner objects
        matches = regex.compile(r"\{(?:[^{}]|(?R))*\}").findall(json_text)
        if len(matches) == 1:
            return _PARSER.parse(matches[0])
        return [_PARSER.parse(mt) for mt in matches]
    except Exception as e:
        raise ValueError("Could not parse JSON from text") from e


def parse_json_llm_plan(llm_text: str) -> GeneratedPlanStep:
    """Parse LLM output into a domain model of the execution model."""
    try:
        parsed_json = llm_text_to_json(llm_text)
        if parsed_json is None:
            raise MisgeneratedPlanException(
                message="The plan cannot be parsed as a JSON",
                output=llm_text,
            )
        # A single step has been generated
        if isinstance(parsed_json, dict):
            parsed_json = [parsed_json]
        if len(parsed_json) == 0 or any(step is None for step in parsed_json):
            raise OutputParserException(error=f"Could not parse JSON out of {llm_text}")
        plan: GeneratedPlanStep = []
        for json_step in parsed_json:
            if "condition" in json_step:
                # Expecting conditional branch

                step = [
                    PlanStep.model_validate(tool) for tool in json_step.get("tools", [])
                ]
                step[0].evaluate_condition = json_step["condition"]
                step[0].raw_plan_text = json.dumps(json_step, ensure_ascii=False)
                if len(json_step) == 0:
                    raise MisgeneratedPlanException(
                        message="Conditional branch should not be empty",
                        output=llm_text,
                    )
            else:
                step = PlanStep.model_validate(json_step)
                step.raw_plan_text = json.dumps(json_step, ensure_ascii=False)
            plan.append(step)
        return plan
    except OutputParserException as e:
        raise MisgeneratedPlanException(
            message=f"Could not parse JSON out of text: {llm_text}",
            output=llm_text,
        ) from e
    except ValidationError as e:
        raise MisgeneratedPlanException(
            message=(
                "Output cannot be parsed as JSON. "
                "Make sure the object is not nested and \"\" are used, not ''"
                f"Failing LLM output: {llm_text}"
            ),
            output=llm_text,
        ) from e
    except ValueError as e:
        raise MisgeneratedPlanException(
            message=f"Could not find a json in text: {llm_text}",
            output=llm_text,
        ) from e


def parse_gml_llm_plan(llm_text: str) -> GeneratedPlanStep:
    """Extract GML output into a domain model of the execution model."""
    plan_graph = _gml_to_nx_graph(llm_text)
    id_to_plan_step: dict[int, PlanStep] = _parse_steps_from_graph(llm_text, plan_graph)
    return _topological_sort_plan_steps(id_to_plan_step, plan_graph)


def _gml_to_nx_graph(llm_text: str) -> nx.Graph:
    """Parse GML output into a networkx graph.

    This allows us to validate the graph of the plan.
    """
    llm_text = llm_text.strip()
    glm_text = re.search(r"graph \[.*\]", llm_text)
    if glm_text is None:
        raise MisgeneratedPlanException(
            message="Plan cannot be parsed as a GML", output=llm_text
        )
    glm_text = glm_text[0].replace("\\n", "").replace("'", "")
    glm_text = glm_text.replace(r"\'", "")
    glm_text = glm_text.replace("]]", "]")
    glm_text = glm_text.replace("[[", "[")
    glm_text = glm_text.replace("\n", "")
    glm_text = glm_text.replace('"]', "")
    glm_text = glm_text.replace('["', "")
    glm_text = glm_text.replace("\\", "")
    # Replace all matches of more than one whitespace with a single space
    glm_text = regex.sub(r"\s{2,}", " ", glm_text)
    try:
        plan_graph: nx.Graph = nx.parse_gml(glm_text, label=None)  # pyright: ignore [reportArgumentType]
    except nx.NetworkXError as e:
        raise MisgeneratedPlanException(
            message="Plan cannot be parsed as a GML",
            output=llm_text,
        ) from e
    if not plan_graph.is_directed():
        raise MisgeneratedPlanException(
            message="GLM plan misses the 'directed 1' directive",
            output=llm_text,
        )
    try:
        _ = nx.find_cycle(plan_graph)
        raise MisgeneratedPlanException(
            message="GLM graph contains cycles",
            output=llm_text,
        )
    except nx.NetworkXNoCycle:
        return plan_graph


def _parse_steps_from_graph(llm_text: str, plan_graph: nx.Graph) -> dict[int, PlanStep]:
    """Convert from the validated networkx graph to a list of PlanSteps.

    In GML each tool call has a numeric ID for the node.
    """
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
    """Sort the plan steps in topological order.

    In GML, order of nodes can be arbitrary, as order is specified by edges.
    GML assumption is that is will be easier to generate syntactically correct
    plans if the order of steps must not be implicitly inferred.
    """
    plan: GeneratedPlanStep = []
    visited: set[int] = set()
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
