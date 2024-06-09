from car_state import CarState
from llm.base import LLMInterface

CAR_STATE = CarState.get_default()

ORACLE: LLMInterface = None
