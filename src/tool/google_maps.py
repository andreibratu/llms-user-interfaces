import requests

from src.configuration import APP_CONFIG
from src.plan.exceptions import BenchmarkException, ExceptionCode
from src.tool import TOOL_CACHE


@TOOL_CACHE.cache
def geocode_address(address: str) -> dict:
    """Geocode an address using the Google Maps API.

    Returns:
        A dictionary with the following keys:
        - latitude: The latitude of the address.
        - longitude: The longitude of the address.
        - address: The formatted address.
        - types: The types of the address.
    """
    response = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={
            "address": address,
            "key": APP_CONFIG.google.maps_api_key,
            "latlng": f"{APP_CONFIG.car_status.latitude},{APP_CONFIG.car_status.longitude}",
        },
    )
    first_match = response.json()["results"][0]
    return {
        "latitude": first_match["geometry"]["location"]["lat"],
        "longitude": first_match["geometry"]["location"]["lng"],
        "address": first_match["formatted_address"],
        "types": first_match["types"],
    }


@TOOL_CACHE.cache
def get_places(query: str, lat: float, lng: float, radius: int | float) -> list[dict]:
    response = requests.post(
        "https://places.googleapis.com/v1/places:searchText",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": APP_CONFIG.google.maps_api_key,
            "X-Goog-FieldMask": (
                "places.displayName,places.formattedAddress,"
                "places.priceLevel,places.menuForChildren,"
                "places.types,places.rating,places.outdoorSeating,"
                "places.goodForGroups,places.evChargeOptions,"
                "places.fuelOptions,places.goodForChildren,"
                "places.outdoorSeating,places.delivery,places.dineIn"
            ),
        },
        json={
            "textQuery": query,
            "maxResultCount": 10,
            "locationBias": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": radius,
                },
            },
        },
    )
    return response.json()["places"]


def distance_matrix(
    origins: list[str] | str,
    destinations: list[str] | str,
) -> dict[str, dict[str, dict[str, str]]]:
    """
    Compute distance and ETA between multiple origins and destinations.

    Args:
        - origins: Multiple origin addresses separated by |.
        - destinations: Multiple destination addresses separated by |.

    Returns:
        A dictionary with the following keys:
        - origin: A dictionary with the following keys:
            - destination: A dictionary with the following keys:
                - distance: The distance between the origin and destination.
                - duration: The estimated time of arrival between the origin and destination.
    """
    response = requests.get(
        "https://maps.googleapis.com/maps/api/distancematrix/json",
        params={
            "origins": ("|".join(origins) if isinstance(origins, list) else origins),
            "destinations": (
                "|".join(destinations)
                if isinstance(destinations, list)
                else destinations
            ),
            "units": "metric",
            "key": APP_CONFIG.google.maps_api_key,
        },
    )

    try:
        all_distances = {}
        rows = response.json()["rows"]
        if len(rows) == 0:
            raise BenchmarkException(
                code=ExceptionCode.INVALID_ARGUMENT,
                message=f"No rows returned by distance_matrix for origins: {origins} and destinations: {destinations}",
            )
        if isinstance(origins, str):
            origins = [origins]
        if isinstance(destinations, str):
            destinations = [destinations]
        for origin, row in zip(origins, rows):
            for dest, element in zip(destinations, row["elements"]):
                if origin not in all_distances:
                    all_distances[origin] = {}
                all_distances[origin][dest] = {
                    "distance": element["distance"]["text"],
                    "duration": element["duration"]["text"],
                }

        return all_distances
    except KeyError as e:
        raise ValueError(
            "Could not locate either one of the origin or destination points"
        ) from e
