from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain import LLMErrorFeedbackStrategyType, PlanFormat


class CarStatus(BaseSettings):
    longitude: float = 11.639028
    latitude: float = 48.251638
    current_address: str = "Parkring 19 85748 Garching bei München"
    home_address: str = "Schröfelhofstraße 20, 81375 München"


class OpenAIConfig(BaseSettings):
    api_key: str | None = None


class GenerationConfig(BaseSettings):
    max_tokens: int = 100
    temperature: float = 1
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.2
    top_p: float = 1
    n: int = 1
    min_machine_instructions_n: int = 2
    rouge_threshold: float = 0.3
    rouge_metrics: list[str] = ["rougeL"]
    timeout_seconds: int = 60
    generate_size: int = 500
    deny_list: list[str] = [
        "{",
        "}",
        "as requested",
        "error",
        "currently",
        "i'm not programmed",
        "...",
        "!",
        "sure",
        "call",
        "the assistant",
        "the scenarios are",
        "your request",
        "once",
        "i can help",
        "i cannot assist",
        "apology",
        "apologies",
        "sorry",
        "i found",
        "your request",
        "miles",
        "mile",
        ":",
        "would you like",
        "-",
        "image",
        "images",
        "write a program",
        "graph",
        "graphs",
        "picture",
        "pictures",
        "file",
        "files",
        "map",
        "maps",
        "draw",
        "plot",
        "go to",
        "car assistant",
    ]
    all_verbs: list[str] = ["VBG", "VB", "VBD", "VBN", "VBP", "VBZ"]
    min_len: int = 20
    # Unlike max tokens, this looks at overall length
    max_len: int = 200


class SpotifyConfig(BaseSettings):
    client_id: str | None = None
    client_secret: str | None = None


class ExperimentConfig(BaseSettings):
    repeat_experiments: int = 1
    wire_producers: list[bool] = [False, True]
    num_demonstrations: list[int] = [1, 5, 10]
    feedback_strategies: "list[LLMErrorFeedbackStrategyType]" = [
        "NO_FEEDBACK",
        "ERROR_TYPE",
        "ERROR_TYPE+STEP",
    ]
    retry_times: list[int] = [1, 3]
    dataset_size: int = 100
    openai_model: str = "gpt-4o-mini-2024-07-18"
    finetune_tool_bert_percentage: float = 1
    finetune_tool_bert_fill_tool_count: int | None = None
    random_seed: int = 42
    max_tool_slice_size: int = 3
    plan_formats: list[PlanFormat] = ["json", "gml"]
    alignment_skip_list: list[str] = [
        "home_address",
        "current_address",
        "current_coordinates",
    ]


class GoogleConfig(BaseSettings):
    maps_api_key: str | None = None
    custom_search_api_key: str | None = None
    custom_search_engine_id: str | None = None


class AppConfig(BaseSettings):
    openai: OpenAIConfig = OpenAIConfig()
    car_status: CarStatus = CarStatus()
    generation: GenerationConfig = GenerationConfig()
    spotify: SpotifyConfig = SpotifyConfig()
    google: GoogleConfig = GoogleConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    retry_max_seconds: int = 60
    timeout_seconds_gde: int = 30

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        str_to_lower=True,
        env_nested_delimiter="__",
    )


APP_CONFIG = AppConfig()
