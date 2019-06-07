from pydantic import BaseModel
from typing import List

_start_list = set(globals())


class LabelledText(BaseModel):
    label: str
    text: str


class Text(BaseModel):
    text: str


class EntityLabel(BaseModel):
    label: str
    token: str
    offset: int


class TextEntities(BaseModel):
    text: str
    entities: List[EntityLabel]


_DOMAIN_DATA_TYPES = {k: v for k, v in globals().items()
                      if k not in _start_list
                      and k[0].isupper()}


def get_domain_type(name):
    return _DOMAIN_DATA_TYPES[name]
