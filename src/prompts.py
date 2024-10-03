from src.llm import LLMMessage
from src.tool.tools import TOOL_SCHEMA

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

DEFAULT_LLM_HYPERS = {
    "temperature": 0,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "top_p": 1,
    "max_tokens": 500,
    "n": 1,
}


_BASE_PLAN_PROMPT_SYS = (
    "You are a car assistant, using tools to satisfy a user query. {tool_schema}"
)

_JSON_LIST_PLAN_SYS = _BASE_PLAN_PROMPT_SYS + (
    " You must output an execution plan using tools above to satisfy user query in JSON format. "
    "Results stored in memory can be called using the pattern $NAME_OF_MEMORY$ anywhere "
    "in other function arguments and the stored value will be used instead of the placeholder."
)

_JSON_LIST_PLAN_W_REASON_SYS = (
    _JSON_LIST_PLAN_SYS
    + "You will give a reason why each tool call was made under the reason field"
)

_GML_PLAN_PROMPT_SYS = (
    _BASE_PLAN_PROMPT_SYS + "You will output the plan using GML direct graph notation. "
    "Each node represent a function call and the edges indicate that "
    "the result of one function call is required in another. "
    "Results stored in memory can be called using the pattern $ID_FUNCTION_CALL$ "
    "anywhere in other function arguments and the stored value will be used "
    "instead of the placeholder. "
)

_GML_PLAN_REASON_PROMPT_SYS = (
    _GML_PLAN_PROMPT_SYS
    + " You will motivate on each node why that function "
    + "call happened using a 'reason' attribute. "
)

_GML_PLAN_REASON_EDGE_PROMPT_SYS = (
    _GML_PLAN_REASON_PROMPT_SYS
    + " You will motivate the existence of each edge using a 'reason' attribute. "
)


def get_system_message(plan_mode: str) -> LLMMessage:
    prompt = {
        "json": _JSON_LIST_PLAN_SYS,
        "json+r": _JSON_LIST_PLAN_W_REASON_SYS,
        "gml": _GML_PLAN_PROMPT_SYS,
        "gml+r": _GML_PLAN_REASON_PROMPT_SYS,
        "gml+r+e": _GML_PLAN_REASON_EDGE_PROMPT_SYS,
    }[plan_mode]
    return LLMMessage(role="system", content=prompt.format(tool_schema=TOOL_SCHEMA))
