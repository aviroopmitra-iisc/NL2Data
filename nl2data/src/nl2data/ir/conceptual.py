"""ConceptualIR model for ER-style conceptual design."""

from typing import List, Literal
from pydantic import BaseModel, Field


class Attribute(BaseModel):
    """An attribute of an entity."""

    name: str
    kind: Literal[
        "identifier",
        "numeric",
        "categorical",
        "text",
        "datetime",
        "boolean",
    ] = "text"


class Entity(BaseModel):
    """An entity in the conceptual model."""

    name: str
    attributes: List[Attribute] = Field(default_factory=list)


class Relationship(BaseModel):
    """A relationship between entities."""

    name: str
    participants: List[str]  # entity names
    cardinality: str = "many_to_one"  # many_to_one | one_to_many | many_to_many


class ConceptualIR(BaseModel):
    """Conceptual ER-style model."""

    entities: List[Entity]
    relationships: List[Relationship] = Field(default_factory=list)

