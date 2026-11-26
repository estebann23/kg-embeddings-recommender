# Validate RDF against the existing SHACL,, and then extract graph records

from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS

from src.kg.models import Entity, Evidence, GraphData, SemanticRelationship, validate_id
from src.kg.semantics import ENTITY_PREFIX, FP, ROOT, domain_labels, relation_definitions, vocabulary


NUTRIENT_PREDICATES = {
    FP.servingSizeGrams: "serving_size_grams",
    FP.calories: "calories",
    FP.totalFatGrams: "total_fat_grams",
    FP.saturatedFatGrams: "saturated_fat_grams",
    FP.cholesterolMilligrams: "cholesterol_milligrams",
    FP.sodiumMilligrams: "sodium_milligrams",
    FP.carbohydrateGrams: "carbohydrate_grams",
    FP.dietaryFiberGrams: "dietary_fiber_grams",
    FP.sugarsGrams: "sugars_grams",
    FP.proteinGrams: "protein_grams",
}


def local_id(uri: URIRef) -> str:
    text = str(uri)
    if not text.startswith(ENTITY_PREFIX):
        if not text.startswith(("http://", "https://")):
            raise ValueError("External entity identifiers must be HTTP IRIs.")
        return "external_" + sha256(text.encode()).hexdigest()[:24]
    identifier = text.removeprefix(ENTITY_PREFIX)
    validate_id(identifier)
    return identifier


def validate_rdf(graph: Graph) -> None:
    conforms, _, report = validate(
        graph, shacl_graph=str(ROOT / "ontology/shapes/food_preferences.shacl.ttl"),
        ont_graph=vocabulary(), inference="none",
    )
    if not conforms:
        raise ValueError("RDF failed SHACL validation:\n" + report)


def extract_graph(graph: Graph) -> GraphData:
    validate_rdf(graph)
    entities = {}
    labels = set(domain_labels())
    for subject, class_uri in graph.subject_objects(RDF.type):
        label = str(class_uri).removeprefix(str(FP))
        if label not in labels:
            continue
        if subject in entities:
            raise ValueError("An entity has multiple domain classes; review before ingestion.")
        names = sorted(graph.objects(subject, RDFS.label), key=lambda value: (value.language != "en", bool(value.language), str(value), value.language or ""))
        props = {
            "label_values": [str(value) for value in names],
            "label_languages": [value.language or "" for value in names],
            "label_datatypes": [str(value.datatype or "") for value in names],
        }
        sources = sorted(str(value) for value in graph.objects(subject, DCTERMS.source))
        if sources:
            props["source"] = sources
        descriptions = sorted(str(value) for value in graph.objects(subject, DCTERMS.description))
        if descriptions:
            props["description"] = descriptions[0]
        if label == "ScenarioProfile":
            props["profile_type"] = "synthetic" if sources and all(value.startswith("urn:fixture:") for value in sources) else "predefined"
        if label == "Ingredient":
            props["canonical_name"] = local_id(subject)
        complete = graph.value(subject, FP.compositionComplete)
        if complete is not None:
            props["composition_complete"] = complete.toPython()
        for predicate, field in NUTRIENT_PREDICATES.items():
            values = list(graph.objects(subject, predicate))
            if values:
                props[field] = float(values[0])
        entities[subject] = Entity(label, local_id(subject), str(names[0]), str(subject), props)
    definitions = relation_definitions()
    by_predicate = {FP[definition.predicate]: definition for definition in definitions.values()}
    observations = defaultdict(list)
    for statement in graph.subjects(RDF.type, RDF.Statement):
        subject = graph.value(statement, RDF.subject)
        predicate = graph.value(statement, RDF.predicate)
        target = graph.value(statement, RDF.object)
        if predicate not in by_predicate or subject not in entities or target not in entities:
            raise ValueError("Statement endpoints or property are outside the validated domain graph.")
        rating = graph.value(statement, FP.ratingValue)
        observations[(subject, predicate, target)].append(Evidence(
            local_id(statement), str(statement), str(graph.value(statement, FP.evidenceState)),
            tuple(sorted(str(value) for value in graph.objects(statement, DCTERMS.source))),
            str(graph.value(statement, FP.sourceText)) if graph.value(statement, FP.sourceText) is not None else None,
            int(rating) if rating is not None else None,
        ))
    relationships = []
    for (subject, predicate, target), records in sorted(observations.items(), key=lambda item: tuple(map(str, item[0]))):
        key = sha256("\n".join(map(str, (subject, predicate, target))).encode()).hexdigest()
        relationships.append(SemanticRelationship(key, str(subject), str(target), by_predicate[predicate].relationship_type,
                                                  tuple(sorted(records, key=lambda record: record.semantic_uri))))
    known_predicates = {RDF.type, RDFS.label, DCTERMS.source, DCTERMS.description, FP.compositionComplete,
                        RDF.subject, RDF.predicate, RDF.object, FP.evidenceState, FP.sourceText,
                        FP.ratingValue} | set(by_predicate) | set(NUTRIENT_PREDICATES)
    for subject, predicate, target in graph:
        if predicate not in known_predicates:
            raise ValueError(f"Unsupported RDF predicate {predicate}; refusing to silently discard data.")
        if predicate == RDF.type and target not in {FP[name] for name in labels} | {RDF.Statement}:
            raise ValueError("Unsupported RDF instance class; review before ingestion.")
    return GraphData(tuple(sorted(entities.values(), key=lambda entity: (entity.label, entity.id))), tuple(relationships))


def load_rdf(path: Path) -> GraphData:
    return extract_graph(Graph().parse(path))
