from typing import Literal

from pydantic import BaseModel


class TypeField(BaseModel):
    type: Literal["string", "number"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeField]
    returns: TypeField


class Prompt(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    name: str
    parameters: dict[str, TypeField]
