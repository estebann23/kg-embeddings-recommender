# Validate all the operational graph semantics against the RDF/SHACL layer

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, XSD

from src.kg.models import GraphData
from src.kg.rdf_loader import NUTRIENT_PREDICATES, extract_graph
from src.kg.semantics import FP, relation_definitions


def to_rdf(data: GraphData) -> Graph:
    """Reconstruct RDF assertions and metadata for operational validation."""
    graph = Graph()
    definitions = relation_definitions()
    for entity in data.entities:
        subject = URIRef(entity.semantic_uri)
        graph.add((subject, RDF.type, FP[entity.label]))
        names = entity.properties.get("label_values", [entity.name])
        languages = entity.properties.get("label_languages", [""] * len(names))
        datatypes = entity.properties.get("label_datatypes", [""] * len(names))
        for name, language, datatype in zip(names, languages, datatypes):
            graph.add((subject, RDFS.label, Literal(name, lang=language or None, datatype=URIRef(datatype) if datatype and not language else None)))
        if "composition_complete" in entity.properties:
            graph.add((subject, FP.compositionComplete, Literal(entity.properties["composition_complete"], datatype=XSD.boolean)))
        for predicate, field in NUTRIENT_PREDICATES.items():
            if field in entity.properties:
                graph.add((subject, predicate, Literal(str(entity.properties[field]), datatype=XSD.decimal)))
        for source in entity.properties.get("source", []):
            graph.add((subject, DCTERMS.source, Literal(source)))
    for edge in data.relationships:
        definition = definitions.get(edge.relationship_type)
        if definition is None:
            raise ValueError("Operational graph contains a noncanonical semantic relationship.")
        subject, predicate, target = URIRef(edge.source_uri), FP[definition.predicate], URIRef(edge.target_uri)
        graph.add((subject, predicate, target))
        for record in edge.observations:
            statement = URIRef(record.semantic_uri)
            for p, value in ((RDF.type, RDF.Statement), (RDF.subject, subject), (RDF.predicate, predicate), (RDF.object, target),
                             (FP.evidenceState, Literal(record.evidence_state))):
                graph.add((statement, p, value))
            for source in record.source:
                graph.add((statement, DCTERMS.source, Literal(source)))
            if record.source_text is not None:
                graph.add((statement, FP.sourceText, Literal(record.source_text)))
            if record.rating is not None:
                graph.add((statement, FP.ratingValue, Literal(record.rating, datatype=XSD.integer)))
    return graph


def validate_operational_graph(repository) -> GraphData:
    data = repository.graph_data()
    if not data.entities:
        raise ValueError("The development graph is empty; load its source RDF first.")
    validate_records(data)
    return data


def validate_records(data: GraphData) -> None:
    checked = extract_graph(to_rdf(data))
    if len(checked.entities) != len(data.entities) or len(checked.relationships) != len(data.relationships):
        raise ValueError("RDF reconstruction lost or duplicated graph entities/relationships.")
    entities = {entity.semantic_uri: entity for entity in checked.entities}
    if any((entity.label, entity.id, entity.name) != (entities[entity.semantic_uri].label, entities[entity.semantic_uri].id, entities[entity.semantic_uri].name)
           for entity in data.entities):
        raise ValueError("Node identity or display name disagrees with its RDF representation.")
    observed = {record.semantic_uri: record.id for edge in checked.relationships for record in edge.observations}
    if any(observed.get(record.semantic_uri) != record.id for edge in data.relationships for record in edge.observations):
        raise ValueError("Evidence record IDs disagree with their statement IRIs.")
    expected_ids = {(edge.source_uri, edge.relationship_type, edge.target_uri): edge.id for edge in checked.relationships}
    if any(edge.id != expected_ids.get((edge.source_uri, edge.relationship_type, edge.target_uri)) for edge in data.relationships):
        raise ValueError("Semantic relationship identities do not match their RDF triples.")
