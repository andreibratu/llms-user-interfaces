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


DEFAULT_ORACLE_LLM_HYPERS = {
    "temperature": 0,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "top_p": 1,
    "max_tokens": 500,
    "n": 1,
}

# TODO: Make hyperparameters part of the benchmark
DEFAULT_PLANNER_LLM_HYPERS = {**DEFAULT_ORACLE_LLM_HYPERS}
