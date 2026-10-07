"""Read semantic definitions from the existing ontology and mapping document."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from rdflib import Graph, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS


ROOT = Path(__file__).resolve().parents[2]
FP = Namespace("https://github.com/estebann23/kg-embeddings-recommender/ontology/food_preferences#")
ENTITY_PREFIX = "https://github.com/estebann23/kg-embeddings-recommender/entity/"


@dataclass(frozen=True)
class RelationDefinition:
    predicate: str
    relationship_type: str
    domains: frozenset[str]
    target: str
    kind: str


@lru_cache(maxsize=1)
def vocabulary() -> Graph:
    return Graph().parse(ROOT / "ontology/food_preferences.ttl")


@lru_cache(maxsize=1)
def domain_labels() -> tuple[str, ...]:
    labels = sorted(str(node).removeprefix(str(FP)) for node in vocabulary().subjects(RDF.type, OWL.Class)
                    if isinstance(node, URIRef) and str(node).startswith(str(FP)))
    if not labels or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", label) for label in labels):
        raise ValueError("Invalid named classes in the domain vocabulary.")
    return tuple(labels)


@lru_cache(maxsize=1)
def relation_definitions() -> dict[str, RelationDefinition]:
    graph = vocabulary()
    mapping = {}
    for line in (ROOT / "ontology/DOMAIN_MODEL.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"(\w+) \| .+ \| ([A-Z_]+) \| .+", line)
        if match:
            if match[1] in mapping:
                raise ValueError("Duplicate RDF mapping in DOMAIN_MODEL.txt.")
            mapping[match[1]] = match[2]
    predicates = {str(node).removeprefix(str(FP)): node for node in graph.subjects(RDF.type, OWL.ObjectProperty)}
    if mapping.keys() != predicates.keys():
        raise ValueError("DOMAIN_MODEL.txt mappings differ from the vocabulary object properties.")
    result = {}
    for name, predicate in predicates.items():
        domain = graph.value(predicate, RDFS.domain)
        union = graph.value(domain, OWL.unionOf)
        domains = set(Collection(graph, union)) if union is not None else {domain}
        target = str(graph.value(predicate, RDFS.range)).removeprefix(str(FP))
        kind = str(graph.value(predicate, FP.constraintKind))
        domain_names = frozenset(str(node).removeprefix(str(FP)) for node in domains)
        if not domain_names <= set(domain_labels()) or target not in domain_labels() or kind not in {"hard", "soft", "fact", "rule"}:
            raise ValueError("Invalid object-property definition.")
        result[mapping[name]] = RelationDefinition(name, mapping[name], domain_names, target, kind)
    return result
