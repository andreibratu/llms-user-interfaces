from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class CarStatus(BaseSettings):
    longitude: float = 11.639028
    latitude: float = 48.251638
    current_address: str = "Parkring 19 85748 Garching bei München"
    home_address: str = "Schröfelhofstraße 20, 81375 München"


class OpenAIConfig(BaseSettings):
    api_key: Optional[str] = None


class GenerationConfig(BaseSettings):
    max_tokens: int = 100
    temperature: float = 1
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.2
    top_p: float = 1
    n: int = 1
    rouge_threshold: float = 0.3
    rouge_metrics: List[str] = ["rougeL"]
    timeout_seconds: int = 60
    generate_size: int = 500
    deny_list: List[str] = [
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
    all_verbs: List[str] = ["VBG", "VB", "VBD", "VBN", "VBP", "VBZ"]
    min_len: int = 20
    # Unlike max tokens, this looks at overall length
    max_len: int = 200


class SpotifyConfig(BaseSettings):
    client_id: Optional[str] = None
    client_secret: Optional[str] = None


class MongoConfig(BaseSettings):
    username: Optional[str] = None
    password: Optional[str] = None
    hostname: str = "localhost"

    @property
    def connect_url(self):
        return f"mongodb://{self.username}:{self.password}@{self.hostname}:27017/"


class ExperimentConfig(BaseSettings):
    repeat_experiments: int = 1
    wire_producers: List[bool] = [False, True]
    num_demonstrations: List[int] = [10]
    feedback_strategies: List[str] = ["ERROR_TYPE+STEP"]
    retry_times: List[int] = [1]
    use_alignment_prediction: List[bool] = [True, False]
    dataset_size: int = 10
    openai_models: List[str] = ["gpt-3.5-turbo-0125"]
    finetune_tool_bert_percentage: float = 1
    finetune_tool_bert_fill_tool_count: Optional[int] = None
    random_seed: int = 42
    max_tool_slice_size: int = 3
    plan_output_mode: List[str] = ["json", "json+r", "gml", "gml+r", "gml+r+e"]


class GoogleConfig(BaseSettings):
    maps_api_key: Optional[str] = None
    custom_search_api_key: Optional[str] = None
    custom_search_engine_id: Optional[str] = None


class AppConfig(BaseSettings):
    openai: OpenAIConfig = OpenAIConfig()
    car_status: CarStatus = CarStatus()
    generation: GenerationConfig = GenerationConfig()
    spotify: SpotifyConfig = SpotifyConfig()
    mongo: MongoConfig = MongoConfig()
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
