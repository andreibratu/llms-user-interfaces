from collections import defaultdict
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from configuration import APP_CONFIG
from domain import Coordinates

DrivingModeType = Literal["personal", "sport", "efficient"]


CarSeats = Literal["all", "driver", "left", "passenger", "right"]


AmbientLightType = Literal[
    "beige", "blue", "cyan", "green", "orange", "red", "white", "yellow"
]


class CarSeatsSchema(BaseModel):
    driver: int
    left: int
    passenger: int
    right: int


class MediaControlSchema(BaseModel):
    now_playing: Optional[str]
    playing: bool
    volume: int


class CarState(BaseModel):
    temperature: CarSeatsSchema
    climate_running: bool
    media_control: MediaControlSchema
    home_address: str
    battery_range: int
    destination_waypoints: List[str]
    window: CarSeatsSchema
    driving_mode: DrivingModeType
    ambient_light: AmbientLightType
    current_coordinates: Coordinates
    current_address: str
    conversations: Dict[str, List[str]] = Field(
        description="A person to text messages map"
    )
    speak: List[str] = []

    @classmethod
    def get_default(cls) -> "CarState":
        return CarState.model_validate(
            {
                "climate_running": True,
                "temperature": {
                    "driver": 20,
                    "left": 20,
                    "passenger": 20,
                    "right": 20,
                },
                "current_address": APP_CONFIG.car_status.current_address,
                "home_address": APP_CONFIG.car_status.home_address,
                "battery_range": 320,
                "media_control": {
                    "playing": False,
                    "now_playing": None,
                    "volume": 25,
                },
                "destination_waypoints": [],
                "window": {
                    "driver": 0,
                    "left": 0,
                    "right": 0,
                    "passenger": 0,
                },
                "range": 320,
                "driving_mode": "personal",
                "ambient_light": "blue",
                "current_coordinates": {
                    "lat": APP_CONFIG.car_status.latitude,
                    "lng": APP_CONFIG.car_status.longitude,
                },
                "fuel_type": "diesel",
                "conversations": defaultdict(list),
                "speak": [],
            }
        )
