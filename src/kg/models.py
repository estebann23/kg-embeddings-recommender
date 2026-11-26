# Small recorss for validarion, by RDF/JSON ingestion and the graph layer

from dataclasses import dataclass, field
import math
import re
from typing import Any
from urllib.parse import urlsplit

from src.kg.schema import DOMAIN_LABELS


def validate_id(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Stable IDs must contain only letters, digits, underscores, or hyphens.")


def validate_iri(value: str) -> None:
    if not isinstance(value, str) or not urlsplit(value).scheme or any(c.isspace() for c in value):
        raise ValueError("Semantic identifiers must be absolute IRIs without whitespace.")


@dataclass(frozen=True)
class Entity:
    label: str
    id: str
    name: str
    semantic_uri: str
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.label not in DOMAIN_LABELS:
            raise ValueError("Unknown domain label.")
        validate_id(self.id)
        validate_iri(self.semantic_uri)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Entities require a nonblank name.")
        reserved = {"id", "name", "semantic_uri", "dataset_id", "managed_by"}
        if reserved & self.properties.keys():
            raise ValueError("Entity properties cannot override identity or dataset ownership.")
        for key, value in self.properties.items():
            values = value if isinstance(value, list) else [value]
            if not isinstance(key, str) or not all(
                isinstance(item, (str, bool, int, float)) and
                (not isinstance(item, float) or math.isfinite(item)) for item in values
            ):
                raise ValueError("Node properties must be finite scalar values or scalar lists.")
        if "composition_complete" in self.properties and (self.label != "Recipe" or not isinstance(self.properties["composition_complete"], bool)):
            raise ValueError("Composition completeness must be a boolean on a Recipe when asserted.")
        if self.label == "ScenarioProfile" and self.properties.get("profile_type", "predefined") not in {"synthetic", "predefined"}:
            raise ValueError("Profiles must be synthetic or predefined at this stage.")

    def as_row(self, dataset_id: str) -> dict[str, Any]:
        validate_id(dataset_id)
        return {
            "id": self.id,
            "properties": {
                **self.properties, "id": self.id, "name": self.name,
                "semantic_uri": self.semantic_uri, "dataset_id": dataset_id,
                "managed_by": "recipe-recommendation-kg",
            },
        }


@dataclass(frozen=True)
class Evidence:
    id: str
    semantic_uri: str
    evidence_state: str
    source: tuple[str, ...]
    source_text: str | None = None
    rating: int | None = None

    def __post_init__(self):
        validate_id(self.id)
        validate_iri(self.semantic_uri)
        if self.evidence_state not in {"CONFIRMED", "INFERRED", "MAY_CONTAIN", "UNKNOWN"}:
            raise ValueError("Invalid evidence state.")
        if not self.source or any(not isinstance(value, str) or not value.strip() for value in self.source):
            raise ValueError("Evidence requires source identifiers/references.")
        if self.source_text is not None and not isinstance(self.source_text, str):
            raise ValueError("Source text must be a string when supplied.")
        if self.rating is not None and (not isinstance(self.rating, int) or isinstance(self.rating, bool) or not 1 <= self.rating <= 5):
            raise ValueError("Observed ratings must be integers in 1..5.")


@dataclass(frozen=True)
class SemanticRelationship:
    id: str
    source_uri: str
    target_uri: str
    relationship_type: str
    observations: tuple[Evidence, ...]


@dataclass(frozen=True)
class GraphData:
    entities: tuple[Entity, ...]
    relationships: tuple[SemanticRelationship, ...]


@dataclass(frozen=True)
class CompatibilityResult:
    profile_id: str
    recipe_id: str
    status: str
    reasons: tuple[dict[str, Any], ...] = ()
