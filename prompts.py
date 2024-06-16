import json

from tool.tools import TOOL_SCHEMA

STATE_SYSTEM_INIT_MSG = (
    "You are an AI assistant managing the state of a car. "
    "You use tools to control the state of the car. "
)

PREDICT_NEXT_STATE_SYSTEM_MSG = (
    f"{STATE_SYSTEM_INIT_MSG}"
    "The tool output can either affect the state of the car or "
    "it can be stored in memory for later use. "
    "The memory and state are JSON strings. "
    "Values taken from memory are put between dollars signs, "
    "e.g. $MEMORY_VARIABLE$. "
    "You are given a list of past actions and how the have affected "
    "the car and memory state. You must predict the next change."
)

PREDICT_FINAL_STATE_SYSTEM_MSG = (
    f"{STATE_SYSTEM_INIT_MSG}"
    "You are given the initial state of the car as a JSON string. "
    "Predict how the state will change given a query. "
    "Keep the JSON schema. If there are values that will be "
    "determined from calling tools, use placeholders. "
)


PREDICT_FINAL_STATE_QUERY_TEMPLATE = (
    "\n"
    "Initial State: {init_state}\n"
    "Query: Change driver seat temperature for driver to 25\n"
    "Final State: {example_one}\n"
    "\n"
    "Initial State: {init_state}\n"
    "Query: Change driving to sport and start a gym playlist\n"
    "Final State: {example_two}\n"
    "Initial State: {init_state}"
    "Query: Change the cockpit to red tell me a joke\n"
    "Final State: {example_three}"
    "Initial State: {init_state}"
    "Query: {query}"
)


DEFAULT_LLM_HYPERS = {
    "temperature": 0,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "top_p": 1,
    "max_tokens": 500,
    "n": 1,
}

BASE_PLAN_PROMPT_SYS = (
    "You are a car assistant, using tools to satisfy a user query. "
    f"The tools you have available are:\n{json.dumps(TOOL_SCHEMA, indent=1)}\n"
)

JSON_LIST_PLAN_SYS = (
    BASE_PLAN_PROMPT_SYS + "Tool calls are represented as JSON objects. Results stored "
    "in memory can be called using the pattern $NAME_OF_MEMORY$ anywhere "
    "in other function arguments and the stored value will be used "
    "instead of the placeholder. "
)

JSON_LIST_PLAN_W_REASON_SYS = (
    JSON_LIST_PLAN_SYS
    + "You will give a reason why each tool call was made under the reason field"
)

GML_PLAN_PROMPT_SYS = (
    BASE_PLAN_PROMPT_SYS + "You will output the plan using GML direct graph notation. "
    "Each node represent a function call and the edges indicate that "
    "the result of one function call is required in another. "
    "Results stored in memory can be called using the pattern $ID_FUNCTION_CALL$ "
    "anywhere in other function arguments and the stored value will be used "
    "instead of the placeholder. "
)

GML_PLAN_REASON_PROMPT_SYS = (
    GML_PLAN_PROMPT_SYS
    + " You will motivate on each node why that function "
    + "call happened using a 'reason' attribute. "
)

GML_PLAN_REASON_EDGE_PROMPT_SYS = (
    GML_PLAN_REASON_PROMPT_SYS
    + " You will motivate the existence of each edge using a 'reason' attribute. "
)


def get_system_prompt(plan_mode: str) -> str:
    return {
        "json": JSON_LIST_PLAN_SYS,
        "json+r": JSON_LIST_PLAN_W_REASON_SYS,
        "gml": GML_PLAN_PROMPT_SYS,
        "gml+r": GML_PLAN_REASON_PROMPT_SYS,
        "gml+r+e": GML_PLAN_REASON_EDGE_PROMPT_SYS,
    }[plan_mode]
