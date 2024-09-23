import datetime
import json
import re
import typing
from typing import list, Literal, Optional, Union

import cachetools.func
import requests
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from timezonefinder import TimezoneFinder

import src.session as SESSION
from src.car_state import AmbientLightType, CarSeats, DrivingModeType
from src.configuration import APP_CONFIG
from src.domain import JSONType
from llm.base import LLMMessage, LLMResponse
from plan.exceptions import ExceptionCode, BenchmarkException
from tool.google_maps import distance_matrix, geocode_address, get_places
from tool.tool_cache import TOOL_CACHE
from tool.websearch import internet_scrape

SpotifySearch = Literal[
    "album", "artist", "playlist", "track", "show", "episode", "audiobook"
]


def set_ambient_light(light_color: AmbientLightType):
    try:
        light_color = light_color.lower()
    except Exception as e:
        raise BenchmarkException(
            tool_name="set_ambient_light",
            taxonomy="illegal_type",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="Could not parse light_color as string",
        ) from e
    allow_list = typing.get_args(AmbientLightType)
    if light_color not in allow_list:
        raise BenchmarkException(
            tool_name="set_ambient_light",
            taxonomy="out_of_enum",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message=(
                "light_color argument should be one "
                f"of the the following: {allow_list}"
            ),
        )
    SESSION.CAR_STATE.ambient_light = light_color


def set_driving_mode(driving_mode: DrivingModeType) -> None:
    allow_list = typing.get_args(DrivingModeType)
    driving_mode = driving_mode.lower()
    if driving_mode not in allow_list:
        raise BenchmarkException(
            tool_name="set_driving_mode",
            taxonomy="out_of_enum",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message=(
                "driving_mode arguments should be one "
                f"of the following: {allow_list}, input was {driving_mode}"
            ),
        )
    SESSION.CAR_STATE.driving_mode = driving_mode


def set_temperature(temperature: int, seatzone: CarSeats) -> None:
    seatzone = seatzone.lower()
    if not 18 <= temperature <= 25:
        raise BenchmarkException(
            tool_name="set_temperature",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="Temperature should be between 18 and 25 inclusive",
        )
    if seatzone not in (cs := typing.get_args(CarSeats)):
        raise BenchmarkException(
            tool_name="set_temperature",
            taxonomy="out_of_enum",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message=f"seatzone parameter should be one of {cs}",
        )
    if seatzone == "all":
        SESSION.CAR_STATE.temperature = SESSION.CAR_STATE.temperature.model_copy(
            update={sz: temperature for sz in cs if sz != "all"}
        )
    else:
        SESSION.CAR_STATE.temperature = SESSION.CAR_STATE.temperature.model_copy(
            update={seatzone: temperature}
        )


def send_text(who: str, what: str):
    if who is None or what is None:
        raise BenchmarkException(
            tool_name="send_text",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="Parameters of send_text should not be null",
        )
    if who not in SESSION.CAR_STATE.conversations:
        SESSION.CAR_STATE.conversations[who] = []
    SESSION.CAR_STATE.conversations[who].append(what)


def set_window(
    window_level: int,
    what_window: CarSeats,
):
    try:
        what_window = what_window.lower()
    except Exception as e:
        raise BenchmarkException(
            tool_name="set_window",
            taxonomy="illegal_type",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="what_window parameter should be a string",
        ) from e
    if not 0 <= window_level <= 100:
        raise BenchmarkException(
            tool_name="set_window",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="window_level should be between 0 and 100 inclusive",
        )
    if what_window not in (cs := typing.get_args(CarSeats)):
        raise BenchmarkException(
            tool_name="set_window",
            taxonomy="out_of_enum",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message=f"what_window value should be one of {cs}",
        )
    if what_window == "all":
        SESSION.CAR_STATE.window = SESSION.CAR_STATE.window.model_copy(
            update={sz: window_level for sz in typing.get_args(CarSeats) if sz != "all"}
        )
    else:
        SESSION.CAR_STATE.window = SESSION.CAR_STATE.window.model_copy(
            update={what_window: window_level}
        )


def media_control_play(playing: bool) -> None:
    if playing not in [True, False]:
        raise BenchmarkException(
            tool_name="media_control_play",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="playing argument value should be boolean",
        )

    SESSION.CAR_STATE.media_control.playing = playing


def media_control_enqueue(media_object: JSONType) -> None:
    if not media_object:
        raise BenchmarkException(
            tool_name="media_control_enqueue",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="media_object argument should not be null",
        )
    SESSION.CAR_STATE.media_control.now_playing = media_object
    SESSION.CAR_STATE.media_control.playing = True


def media_control_volume(volume: int) -> None:
    try:
        volume = int(volume)
    except TypeError as e:
        raise BenchmarkException(
            tool_name="media_control_volume",
            taxonomy="illegal_type",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="Cannot parse volume into an int",
        ) from e
    if volume not in range(0, 101):
        # Should also catch type errors
        raise BenchmarkException(
            tool_name="media_control_volume",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message=(
                "volume should be in the 0 (mute) - 100 (max volume) range inclusive"
            ),
        )
    SESSION.CAR_STATE.media_control.volume = volume
    if volume > 0:
        SESSION.CAR_STATE.media_control.playing = True


def get_current_date() -> str:
    timezone = TimezoneFinder().timezone_at(
        lat=SESSION.CAR_STATE.current_coordinates.lat,
        lng=SESSION.CAR_STATE.current_coordinates.lng,
    )
    now = datetime.datetime.now()
    time_string = now.strftime("%A, %d %B %Y, %H:%M")
    return f"{time_string} {timezone}"


@TOOL_CACHE.cache
def places_tool(
    query: str,
    location: Optional[str] = None,
    search_range: Optional[float] = None,
) -> JSONType:
    if search_range is None:
        search_range = 20

    if query is None:
        raise BenchmarkException(
            tool_name="places_tool",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="query argument cannot be null",
        )

    try:
        search_range = max(float(search_range), 50)
    except (TypeError, ValueError) as e:
        raise BenchmarkException(
            tool_name="places_tool",
            taxonomy="illegal_type",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message=("search_range could not be parsed into a float"),
        ) from e

    if location is None:
        lat = SESSION.CAR_STATE.current_coordinates.lat
        lng = SESSION.CAR_STATE.current_coordinates.lng
    else:
        response = geocode_address(location)
        lat = response["latitude"]
        lng = response["longitude"]
    radius = search_range * 1000

    return json.dumps(get_places(query, lat, lng, radius))


@TOOL_CACHE.cache
def search_internet(query: str) -> str:
    try:
        return internet_scrape(query)
    except Exception as e:
        raise BenchmarkException(
            "search_internet",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            taxonomy="illegal_state",
            message=str(e),
        ) from e


def speak_user(query: str, data: Optional[JSONType] = None) -> None:
    if data is None:
        data = ""
    if not isinstance(query, str) or not isinstance(data, JSONType):
        raise BenchmarkException(
            tool_name="speak_user",
            taxonomy="illegal_type",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="Arguments for speak_user should be strings or JSON objects",
        )
    response: LLMResponse = SESSION.LLM.invoke(
        [
            LLMMessage(
                role="system",
                content=(
                    "You role play as a voice assistant in a car. "
                    "You must give very short answers to the driver."
                ),
            ),
            LLMMessage(role="system", content=query),
            LLMMessage(role="user", content=data),
        ]
    )
    if response is None:
        # TODO: fix false flag safety from Azure OpenAI
        SESSION.CAR_STATE.speak.append(f"{query} {data}")
    else:
        SESSION.CAR_STATE.speak.append(response.text)


def set_navigation(waypoints: Union[list[str], str]) -> JSONType:
    if isinstance(waypoints, str):
        try:
            # Might be stringified when stored in memory
            waypoints = json.loads(waypoints)
        except json.JSONDecodeError:
            # Leave value as it is
            waypoints = [waypoints]
    SESSION.CAR_STATE.destination_waypoints = waypoints
    try:
        return distance_matrix(SESSION.CAR_STATE.current_address, waypoints)
    except ValueError as e:
        raise BenchmarkException(
            tool_name="set_navigation",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="Unknown origin or destination - are they proper addresses?",
        ) from e


def add_navigation(waypoints: Union[list[str], str]) -> JSONType:
    if isinstance(waypoints, str):
        try:
            # Might be stringified when stored in memory
            waypoints = json.loads(waypoints)
        except json.JSONDecodeError:
            # Leave value as it is
            waypoints = [waypoints]
    SESSION.CAR_STATE.destination_waypoints.extend(waypoints)
    try:
        return distance_matrix(SESSION.CAR_STATE.current_address, waypoints)
    except ValueError as e:
        raise BenchmarkException(
            tool_name="set_navigation",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="Unknown origin or destination - are they proper addresses?",
        ) from e


def get_car_state() -> JSONType:
    return SESSION.CAR_STATE.model_dump_json()


@TOOL_CACHE.cache
def query_to_place(query: str) -> JSONType:
    return json.dumps(geocode_address(query))


@retry(
    retry=retry_if_not_exception_type((BenchmarkException, TypeError)),
    wait=wait_random_exponential(min=1, max=APP_CONFIG.retry_max_seconds),
    stop=stop_after_attempt(5),
)
def weather_tool(query: Optional[str] = None):
    if query is None:
        lat = SESSION.CAR_STATE.current_coordinates.lat
        lng = SESSION.CAR_STATE.current_coordinates.lng
    else:
        geocode_response = geocode_address(query)
        lat = geocode_response["latitude"]
        lng = geocode_response["longitude"]

    timezone = TimezoneFinder().timezone_at(lat=lat, lng=lng)

    weather_code_mapping = {
        0: "clear_sky",
        1: "few_clouds",
        2: "partly_cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing_rime_fog",
        51: "light_drizzle",
        53: "moderate_drizzle",
        55: "heavy_drizzle",
        56: "light_freezing_drizzle",
        57: "heavy_freezing_drizzle",
        61: "light_rain",
        63: "moderate_rain",
        65: "heavy_rain",
        66: "light_freezing_rain",
        67: "heavy_freezing_rain",
        71: "light_snow",
        73: "moderate_snow",
        75: "heavy_snow",
        77: "snow_grains",
        80: "showers",
        81: "modern_showers",
        82: "violent_showers",
        85: "slight_snow_shower",
        86: "heavy_snow_shower",
        95: "thunderstorm",
        96: "thunderstorm_slight_hail",
        97: "thunderstorm_heavy_hail",
    }

    query_params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": (
            "apparent_temperature,precipitation_probability,precipitation,"
            "weather_code,cloud_cover,snowfall"
        ),
        "daily": ("weather_code,sunrise,sunset"),
        "timezone": timezone,
        "forecast_days": 3,
    }
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=query_params,
    )
    report = response.json()
    # Discard redundant keys
    for k in [
        "latitude",
        "longitude",
        "timezone",
        "generationtime_ms",
        "utc_offset_seconds",
        "timezone_abbreviation",
        "current_units",
        "daily_units",
    ]:
        if k in report:
            del report[k]
    for k, values in report["hourly"].items():
        # Discard hours to keep the report token efficient
        report["hourly"][k] = values[::6]
    for time_period in ["hourly", "daily"]:
        report[time_period]["weather_code"] = [
            weather_code_mapping.get(code, code)
            for code in report[time_period]["weather_code"]
        ]
    return json.dumps(report)


def llm_parse_json(
    query: str, json_object: str, return_type: Literal["object", "string"]
) -> Union[JSONType, str]:
    if return_type not in ["object", "string"]:
        raise BenchmarkException(
            tool_name="llm_parse_json",
            taxonomy="out_of_enum",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="return_type should be one of 'object', 'string' or 'list'",
        )
    sub_prompt = {
        "object": (
            "Return a JSON object or list of JSON objects. "
            "Do not nest it or them under any key."
        ),
        "string": (
            "Return a single string, found inside JSON "
            "object. Only return the value, do not "
            "prepend it or introduce it in any way."
        ),
    }[return_type]
    prompt = (
        "You are receiving a JSON input and are required to apply an operation to it. "
        + sub_prompt
    )
    result: LLMResponse = SESSION.LLM.invoke(
        [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content=f"Query: {query}\nObject: {json_object}"),
        ],
        temperature=0,
    )
    if result.text == "":
        raise BenchmarkException(
            tool_name="llm_parse_json",
            taxonomy="illegal_value",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message="Empty LLM response",
        )
    output = result.text

    if return_type == "string":
        mt = re.search(r"\"(.+)\"", result.text)
        if mt:
            output = mt[1]

    try:
        # LLM has a tendency to nest values under one key
        # do some post-processing
        json_output = json.loads(output)
        if isinstance(json_output, dict):
            if len(json_output) == 1:
                output = [list(json_output.values())[0]]
        if isinstance(json_output, list):
            if all(isinstance(item, dict) and len(item) == 1 for item in json_output):
                output = [list(item.values())[0] for item in json_output]
        output = json.dumps(output)
    except json.JSONDecodeError:
        pass

    return output


@cachetools.func.ttl_cache(ttl=55 * 60)  # 55 minutes
def _get_access_token():
    """Returns an access available that lasts 60 minutes."""
    response = requests.post(
        url="https://accounts.spotify.com/api/token",
        data={
            "grant_type": "client_credentials",
            "client_id": APP_CONFIG.spotify.client_id,
            "client_secret": APP_CONFIG.spotify.client_secret,
        },
        timeout=5,
    )
    return response.json()["access_token"]


@TOOL_CACHE.cache
@retry(
    retry=retry_if_not_exception_type((BenchmarkException, TypeError)),
    wait=wait_random_exponential(min=1, max=APP_CONFIG.retry_max_seconds),
    stop=stop_after_attempt(5),
)
def media_search(query: str, types: list[SpotifySearch]) -> JSONType:
    if any(tf not in (ss := typing.get_args(SpotifySearch)) for tf in types):
        not_allowed = [wrong_type for wrong_type in types if wrong_type not in ss]
        raise BenchmarkException(
            tool_name="media_search",
            taxonomy="out_of_enum",
            code=ExceptionCode.ARGUMENT_VALIDATION,
            message=(
                f"types parameter values should all be from {ss}, "
                f"the following are illegal: {not_allowed}"
            ),
        )

    response = requests.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {_get_access_token()}"},
        params={"q": query, "type": ",".join(types), "limit": 2},
    )

    return response.json()


@TOOL_CACHE.cache
def truthy(query: str) -> bool:
    response: LLMResponse = SESSION.LLM.invoke(
        [
            LLMMessage(
                role="system",
                content=(
                    "You are receiving a query and are required to "
                    "determine if it is true or false."
                ),
            ),
            LLMMessage(role="user", content=query),
        ],
        temperature=0,
    )
    return response.text.lower() in [
        "yes",
        "true",
        "correct",
        "right",
    ]


TOOL_SCHEMA = [
    {
        "name": "set_temperature",
        "role": "consumer",
        "description": "Set temperature in car cockpit for each passenger individually",
        "args": {
            "temperature": {"required": True, "type": "integer", "min": 18, "max": 25},
            "seatzone": {
                "required": True,
                "type": "string",
                "enum": typing.get_args(CarSeats),
            },
        },
    },
    {
        "name": "set_window",
        "role": "consumer",
        "description": "Control car windows for each passenger individually",
        "args": {
            "window_level": {
                "required": True,
                "type": "integer",
                "min": 0,
                "max": 100,
                "description": "100 is fully open, 0 fully closed",
            },
            "what_window": {
                "required": True,
                "type": "string",
                "enum": typing.get_args(CarSeats),
            },
        },
    },
    {
        "name": "set_ambient_light",
        "role": "consumer",
        "description": "Set ambient light in car cockpit",
        "args": {
            "light_color": {
                "required": True,
                "type": "string",
                "enum": typing.get_args(AmbientLightType),
            }
        },
    },
    {
        "name": "set_driving_mode",
        "role": "consumer",
        "description": "Set car driving mode",
        "args": {
            "driving_mode": {
                "required": True,
                "type": "string",
                "enum": typing.get_args(DrivingModeType),
            }
        },
    },
    {
        "name": "send_text",
        "role": "consumer",
        "description": "Send message to a person in agenda",
        "args": {
            "who": {
                "required": True,
                "type": "string",
            },
            "what": {
                "required": True,
                "type": "string",
            },
        },
    },
    {
        "name": "media_control_play",
        "role": "consumer",
        "description": (
            "Pauses or resumes media system playing"
            "A media object should be selected first through media_control_enqueue"
        ),
        "args": {
            "playing": {
                "required": True,
                "type": "boolean",
            }
        },
    },
    {
        "name": "media_control_enqueue",
        "role": "consumer",
        "description": "Play audio in car system. Use with media_search returns.",
        "args": {"media_object": {"required": True, "type": "object"}},
    },
    {
        "name": "media_control_volume",
        "role": "consumer",
        "description": "Set car media volume",
        "args": {
            "volume": {
                "required": True,
                "type": "integer",
                "min": 0,
                "max": 100,
                "description": "100 is max, 0 is mute",
            },
        },
    },
    {
        "name": "get_current_date",
        "role": "producer",
        "description": "Return current datetime of the car",
        "args": None,
    },
    {
        "name": "places_tool",
        "role": "producer",
        "description": (
            "Returns geographical points of interest. "
            "Information can be extracted using llm_parse_json tool. "
            "Maximum search range is 50 kilometers, but relevant results outside "
            "might be returned by the API."
        ),
        "args": {
            "query": {
                "required": True,
                "type": "string",
            },
            "location": {
                "required": True,
                "type": "string",
                "description": (
                    "Constrains searching only for place around "
                    "given address. Uses current car location if null"
                ),
            },
            "search_range": {
                "required": False,
                "type": "integer",
                "description": "Search range in kilometers. Default 20, maximum 50",
            },
        },
    },
    {
        "name": "search_internet",
        "role": "producer",
        "description": "Search the internet for information",
        "args": {
            "query": {
                "required": True,
                "type": "string",
                "description": "What to search",
            }
        },
    },
    {
        "name": "speak_user",
        "role": "consumer",
        "description": (
            "Speak to driver. Only use if the user explicitly demands information from you."
        ),
        "args": {
            "query": {
                "required": True,
                "type": "string",
                "description": "What to do with data",
            },
            "data": {
                "required": False,
                "type": "object",
                "description": (
                    "Data to interpret for the user. Can "
                    "be null if query needs no data"
                ),
            },
        },
    },
    {
        "name": "set_navigation",
        "role": "consumer",
        "description": (
            "Set car navigation to one or more waypoints. Returns ETAs "
            + "to each of them from current location"
        ),
        "args": {
            "waypoints": {
                "required": True,
                "type": "list[string]",
                "description": "lists of addresses",
            }
        },
    },
    {
        "name": "add_navigation",
        "role": "consumer",
        "description": (
            "Add new waypoints to car navigation at the end. "
            "Returns ETAs to each of them from current location"
        ),
        "args": {
            "waypoints": {
                "required": True,
                "type": "list[string]",
                "description": "lists of addresses",
            }
        },
    },
    {
        "name": "get_car_state",
        "role": "producer",
        "description": "Get current car state",
        "args": None,
    },
    {
        "name": "query_to_place",
        "role": "producer",
        "description": "Translate query to geocoordinates. Return lat, lng and address.",
        "args": {"query": {"required": True, "type": "string"}},
    },
    {
        "name": "weather_tool",
        "role": "producer",
        "description": "Get weather for location",
        "args": {"query": {"required": True, "type": "string"}},
    },
    {
        "name": "llm_parse_json",
        "role": "logic",
        "description": "Extract information from another tool output",
        "args": {
            "query": {
                "required": True,
                "type": "string",
                "description": "What data to extract from provided data",
            },
            "json_object": {
                "required": True,
                "type": "object",
                "description": "Data to apply query on",
            },
            "return_type": {
                "required": True,
                "type": "string",
                "enum": ["json", "string"],
            },
        },
    },
    {
        "name": "media_search",
        "role": "producer",
        "description": "Search audiobooks, songs, albums, etc.",
        "args": {
            "query": {"required": True, "type": "string"},
            "types": {
                "required": True,
                "type": "string",
                "enum": typing.get_args(SpotifySearch),
            },
        },
    },
    {
        "name": "truthy",
        "role": "",
        "description": "Decide if the answer to a query is true or false",
        "args": {
            "query": {
                "required": True,
                "type": "string",
            }
        },
    },
]

TOOL_NAMES = [tool["name"] for tool in TOOL_SCHEMA]

TOOL_HEADERS = {tool["name"]: tool["description"] for tool in TOOL_SCHEMA}
