# constraints/indexes using the established RDF class names
import logging

from src.kg.semantics import domain_labels


DOMAIN_LABELS = domain_labels()


def initialize_schema(connection) -> None:
    for label in DOMAIN_LABELS:
        connection.write(f"CREATE CONSTRAINT recipe_{label.lower()}_identity IF NOT EXISTS FOR (n:{label}) REQUIRE (n.dataset_id, n.id) IS UNIQUE")
    connection.write("CREATE CONSTRAINT recipe_evidence_identity IF NOT EXISTS FOR (n:EvidenceRecord) REQUIRE (n.dataset_id, n.id) IS UNIQUE")
    connection.write("CREATE CONSTRAINT recipe_semantic_uri IF NOT EXISTS FOR (n:RecipeRecommendationKG) REQUIRE (n.dataset_id, n.semantic_uri) IS UNIQUE")
    for label, field in (("Ingredient", "canonical_name"), ("Recipe", "name")):
        connection.write(f"CREATE INDEX recipe_{label.lower()}_{field} IF NOT EXISTS FOR (n:{label}) ON (n.{field})")
    connection.write("CREATE INDEX recipe_evidence_relationship IF NOT EXISTS FOR (n:EvidenceRecord) ON (n.relationship_id)")
    logging.getLogger(__name__).info("Schema initialized")
