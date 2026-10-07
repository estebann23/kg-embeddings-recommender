from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json

from src.kg.rdf_loader import load_rdf
from src.kg.semantics import ENTITY_PREFIX, FP, ROOT, relation_definitions


@lru_cache(maxsize=1)
def seed_data():
    return load_rdf(ROOT / "data/synthetic/seed.ttl")


def expected():
    return json.loads((ROOT / "data/synthetic/expected.json").read_text(encoding="utf-8"))


def change_evidence(data, source_id, relationship_type, target_id, state):
    lookup = {entity.semantic_uri: entity.id for entity in data.entities}
    changed = []
    for edge in data.relationships:
        if lookup[edge.source_uri] == source_id and edge.relationship_type == relationship_type and lookup[edge.target_uri] == target_id:
            edge = replace(edge, observations=tuple(replace(record, evidence_state=state) for record in edge.observations))
        changed.append(edge)
    return replace(data, relationships=tuple(changed))


def rewire_relationship(data, source_id, relationship_type, target_id, *, new_source_id=None, new_target_id=None):
    by_id = {entity.id: entity for entity in data.entities}
    definition = relation_definitions()[relationship_type]
    changed = []
    for edge in data.relationships:
        if edge.source_uri == by_id[source_id].semantic_uri and edge.relationship_type == relationship_type and edge.target_uri == by_id[target_id].semantic_uri:
            source = by_id[new_source_id or source_id].semantic_uri
            target = by_id[new_target_id or target_id].semantic_uri
            key = sha256("\n".join((source, str(FP[definition.predicate]), target)).encode()).hexdigest()
            observations = tuple(replace(record, id=record.id + "_variant", semantic_uri=ENTITY_PREFIX + record.id + "_variant") for record in edge.observations)
            edge = replace(edge, id=key, source_uri=source, target_uri=target, observations=observations)
        changed.append(edge)
    return replace(data, relationships=tuple(changed))
