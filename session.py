from car_state import CarState
from configuration import AppConfig
from llm.base import LLMInterface

CAR_STATE = CarState.get_default()

# Global reference to LLM under test - used in meta tools, generation
# and other places where the LLM in executor cannot be easily referenced
LLM: LLMInterface = None

APP_CONFIG = AppConfig()
