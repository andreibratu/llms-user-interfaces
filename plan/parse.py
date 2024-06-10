import json
import re
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx
import regex
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import ValidationError

from plan.domain import PlanStep, PlanType
from plan.exceptions import MisgeneratedPlanException

_PARSER = JsonOutputParser()


def llm_text_to_json(llm_text: str) -> Optional[Union[List[Dict], Dict]]:
    json_text = llm_text.replace("'", '"').replace("\n'", "")
    try:
        # Try to discard text around the JSON output of interest
        matches = regex.compile(r"\{(?:[^{}]|(?R))*\}").findall(json_text)
        if len(matches) == 1:
            return _PARSER.parse(matches[0])
        return [_PARSER.parse(mt) for mt in matches]
    except Exception as e:
        raise ValueError("Could not parse JSON from text") from e


def parse_gml_llm_plan(llm_text: str, tokens: int = 0) -> PlanType:
    llm_text = llm_text.strip()
    glm_text = re.search(r"graph \[.*\]", llm_text)
    if glm_text is None:
        raise MisgeneratedPlanException(
            code=24,
            message="Plan cannot be parsed as a GML",
            output=llm_text,
            tokens=tokens,
        )
    try:
        plan_graph: nx.Graph = nx.parse_gml(llm_text, label=None)
    except nx.NetworkXError as e:
        raise MisgeneratedPlanException(
            code=24,
            message="Plan cannot be parsed as a GML",
            output=llm_text,
            tokens=tokens,
        ) from e
    if not plan_graph.is_directed():
        raise MisgeneratedPlanException(
            code=46,
            message="GLM plan misses the 'directed 1' directive",
            output=llm_text,
            tokens=tokens,
        )
    try:
        _ = nx.find_cycle(plan_graph)
        raise MisgeneratedPlanException(
            code=47, message="GLM graph contains cycles", output=llm_text, tokens=tokens
        )
    except nx.NetworkXNoCycle:
        id_to_planstep: Dict[int, PlanStep] = {}
        for node_id in nx.topological_sort(plan_graph):
            raw_node_text = re.search(rf"node \[ id {node_id} .+? \]", llm_text)
            assert raw_node_text, "A node included in the plan should be found text"
            # pylint: disable=protected-access
            node_metadata: Dict[str, Any] = plan_graph._node[node_id]
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

            id_to_planstep[node_id] = PlanStep(
                tool_name=node_metadata["function"],
                args=args,
                raw_plan_text=raw_node_text.group(),
                memory=str(node_id),
                reason=node_metadata.get("reason"),
            )
        plan: PlanType = []
        visited: Set[int] = set()
        for node_id in nx.topological_sort(plan_graph):
            if node_id in visited:
                continue
            visited.add(node_id)
            plan.append(id_to_planstep[node_id])
            for _, nnode, edge_metadata in plan_graph.edges(node_id, data=True):
                if "condition" in edge_metadata:
                    # Gather all nodes from the conditional branch and put them in a sublist
                    branch_first_step = id_to_planstep[nnode]
                    branch_first_step.evaluate_condition = edge_metadata["condition"]
                    branch = [branch_first_step]
                    visited.add(nnode)
                    for _, branch_nnode_id in nx.dfs_edges(plan_graph, nnode):
                        visited.add(branch_nnode_id)
                        branch.append(id_to_planstep[branch_nnode_id])
                    plan.append(branch)
        return plan


def parse_json_llm_plan(llm_text: str, tokens: int = 0) -> PlanType:
    try:
        parsed_json = llm_text_to_json(llm_text)
        if parsed_json is None:
            raise MisgeneratedPlanException(
                code=24,
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
                if len(json_step) == 0:
                    raise MisgeneratedPlanException(
                        code=48,
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
            code=44,
            message="Could not parse JSON out of text",
            output=llm_text,
            tokens=tokens,
        ) from e
    except ValidationError as e:
        raise MisgeneratedPlanException(
            code=25,
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
            code=36,
            message="Could not find a json in text",
            output=llm_text,
            tokens=tokens,
        ) from e
