from typing import Dict, Literal, Union

from pydantic import BaseModel

UserId = str

Person = str

JSONType = str

Metadata = Dict[str, Union[dict, list, str, int, float]]

PlanFormat = Literal["json", "json+r", "gml", "gml+r", "gml+r+e"]


class Coordinates(BaseModel):
    lat: float
    lng: float

    def __hash__(self) -> int:
        return hash((self.lat, self.lng)) + hash(str(type(self)))
