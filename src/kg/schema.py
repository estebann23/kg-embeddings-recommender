"""Idempotent constraints/indexes using the established RDF class names."""
import logging

from src.kg.semantics import domain_labels


DOMAIN_LABELS = domain_labels()


def initialize_schema(connection) -> None:
    for label in DOMAIN_LABELS:
        connection.write(f"CREATE CONSTRAINT fp_{label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
    connection.write("CREATE CONSTRAINT fp_evidence_id IF NOT EXISTS FOR (n:EvidenceRecord) REQUIRE n.id IS UNIQUE")
    connection.write("CREATE CONSTRAINT fp_semantic_uri IF NOT EXISTS FOR (n:FoodPreferenceKG) REQUIRE n.semantic_uri IS UNIQUE")
    for label, field in (("Ingredient", "canonical_name"), ("Dish", "name"), ("Restaurant", "name")):
        connection.write(f"CREATE INDEX fp_{label.lower()}_{field} IF NOT EXISTS FOR (n:{label}) ON (n.{field})")
    connection.write("CREATE INDEX fp_evidence_relationship IF NOT EXISTS FOR (n:EvidenceRecord) ON (n.relationship_id)")
    logging.getLogger(__name__).info("Schema initialized")
