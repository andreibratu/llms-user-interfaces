from src.car_state import CarState
from src.llm import LLMInterface

CAR_STATE: CarState = CarState.get_default()

# Global reference to LLM under test - used in meta tools, generation
# and other places where the LLM in executor cannot be easily referenced
LLM: LLMInterface | None = None
