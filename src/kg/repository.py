"""Atomic RDF-derived upserts and scoped graph reads; no ranking logic."""
from collections import defaultdict
import logging

from src.kg import queries
from src.kg.models import Entity, Evidence, GraphData, SemanticRelationship, validate_id
from src.kg.semantics import FP, relation_definitions


MANAGED_BY = "food-preference-kg"


class GraphRepository:
    def __init__(self, connection, dataset_id: str = "synthetic_v1"):
        validate_id(dataset_id)
        self.connection = connection
        self.dataset_id = dataset_id

    def load(self, data: GraphData) -> None:
        """Upsert one validated batch atomically; retain previous observations."""
        from src.kg.validation import validate_records
        validate_records(data)
        definitions = relation_definitions()
        entities = {entity.semantic_uri: entity for entity in data.entities}
        if len(entities) != len(data.entities):
            raise ValueError("Duplicate entity IRIs in import batch.")
        evidence_uris = set()
        for relationship in data.relationships:
            definition = definitions.get(relationship.relationship_type)
            if definition is None or relationship.source_uri not in entities or relationship.target_uri not in entities:
                raise ValueError("Unknown relationship type or missing endpoint.")
            if entities[relationship.source_uri].label not in definition.domains or entities[relationship.target_uri].label != definition.target:
                raise ValueError("Relationship endpoint types disagree with RDF domain/range.")
            if not relationship.observations:
                raise ValueError("Relationships require evidence.")
            for observation in relationship.observations:
                if observation.semantic_uri in evidence_uris or observation.semantic_uri in entities:
                    raise ValueError("An evidence IRI cannot identify multiple graph records.")
                evidence_uris.add(observation.semantic_uri)
                if (observation.weight is not None) != (definition.kind == "soft"):
                    raise ValueError("Only soft relationships require weights.")
                if observation.evidence_state == "MAY_CONTAIN" and definition.predicate not in {"contains", "hasAllergen"}:
                    raise ValueError("MAY_CONTAIN is only ingredient/allergen evidence.")
        self.connection.write_transaction(self._load_transaction, data)
        logging.getLogger(__name__).info("Loaded %s entities and %s semantic relationships into %s", len(data.entities), len(data.relationships), self.dataset_id)

    def _load_transaction(self, tx, data: GraphData):
        node_groups = defaultdict(list)
        for entity in data.entities:
            node_groups[entity.label].append(entity.as_row(self.dataset_id))
        for label, rows in node_groups.items():
            collisions = tx.run(
                f"UNWIND $rows AS row MATCH (n:{label} {{id: row.id}}) "
                "WHERE coalesce(n.managed_by, '') <> $owner OR coalesce(n.dataset_id, '') <> $dataset_id "
                "OR n.semantic_uri <> row.properties.semantic_uri RETURN n.id AS id",
                rows=rows, owner=MANAGED_BY, dataset_id=self.dataset_id,
            ).data()
            if collisions:
                raise ValueError("Existing entity identity belongs to a different import; no data was changed.")
            tx.run(f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n:FoodPreferenceKG SET n = row.properties",
                   rows=rows).consume()
        relationship_groups = defaultdict(list)
        evidence_rows = []
        definitions = relation_definitions()
        for relationship in data.relationships:
            definition = definitions[relationship.relationship_type]
            relationship_groups[relationship.relationship_type].append({
                "id": relationship.id, "source_uri": relationship.source_uri,
                "target_uri": relationship.target_uri,
                "properties": {"id": relationship.id, "dataset_id": self.dataset_id,
                               "managed_by": MANAGED_BY, "rdf_predicate": str(FP[definition.predicate]), "constraint_kind": definition.kind},
            })
            for observation in relationship.observations:
                properties = {
                    "id": observation.id, "semantic_uri": observation.semantic_uri,
                    "relationship_id": relationship.id, "source_uri": relationship.source_uri,
                    "target_uri": relationship.target_uri, "rdf_predicate": str(FP[definition.predicate]),
                    "evidence_state": observation.evidence_state, "source": list(observation.source),
                    "dataset_id": self.dataset_id, "managed_by": MANAGED_BY,
                }
                if observation.source_text is not None:
                    properties["source_text"] = observation.source_text
                if observation.weight is not None:
                    properties["weight"] = observation.weight
                evidence_rows.append({"id": observation.id, "properties": properties})
        collisions = tx.run(
            "UNWIND $rows AS row MATCH (e:EvidenceRecord {id: row.id}) "
            "WHERE coalesce(e.managed_by, '') <> $owner OR coalesce(e.dataset_id, '') <> $dataset_id "
            "OR e.semantic_uri <> row.properties.semantic_uri OR e.relationship_id <> row.properties.relationship_id RETURN e.id AS id",
            rows=evidence_rows, owner=MANAGED_BY, dataset_id=self.dataset_id,
        ).data()
        if collisions:
            raise ValueError("An evidence identity belongs to a different relationship/import.")
        tx.run("UNWIND $rows AS row MERGE (e:EvidenceRecord {id: row.id}) SET e:FoodPreferenceKG SET e = row.properties",
               rows=evidence_rows).consume()
        for relationship_type, rows in relationship_groups.items():
            count = tx.run(
                "UNWIND $rows AS row MATCH (s:FoodPreferenceKG {semantic_uri: row.source_uri}) "
                "WITH row, s MATCH (t:FoodPreferenceKG {semantic_uri: row.target_uri}) "
                f"MERGE (s)-[r:{relationship_type} {{id: row.id}}]->(t) SET r = row.properties RETURN count(r) AS count",
                rows=rows,
            ).single()["count"]
            if count != len(rows):
                raise ValueError("Relationship endpoint lookup failed; rolling back the import.")
        tx.run(queries.SUMMARIZE_EVIDENCE, dataset_id=self.dataset_id).consume()

    def graph_data(self) -> GraphData:
        """Read nodes, direct edges, and all supporting observations together."""
        # One transaction avoids per-record sessions. Run development imports
        # separately from reads; Neo4j's read-committed isolation is not a snapshot guarantee.
        def read(tx):
            parameters = {"dataset_id": self.dataset_id}
            return (tx.run(queries.NODES, parameters).data(), tx.run(queries.RELATIONSHIPS, parameters).data(),
                    tx.run(queries.EVIDENCE, parameters).data())
        with self.connection.driver.session(database=self.connection.database) as session:
            node_rows, relationship_rows, evidence_rows = session.execute_read(read)
        relationships_by_id = {row["properties"]["id"]: row for row in relationship_rows}
        if len(relationships_by_id) != len(relationship_rows):
            raise ValueError("Duplicate semantic relationship IDs in the operational graph.")
        definitions = relation_definitions()
        for row in relationship_rows:
            definition = definitions.get(row["relationship_type"])
            if definition is None or row["properties"].get("rdf_predicate") != str(FP[definition.predicate]) or row["properties"].get("constraint_kind") != definition.kind:
                raise ValueError("Operational relationship metadata disagrees with the RDF vocabulary.")
        observations = defaultdict(list)
        for row in evidence_rows:
            values = row["properties"]
            relationship = relationships_by_id.get(values["relationship_id"])
            if relationship is None:
                raise ValueError("An orphan evidence record needs review; its semantic relationship is missing.")
            if values["source_uri"] != relationship["source_uri"] or values["target_uri"] != relationship["target_uri"] or values["rdf_predicate"] != relationship["properties"]["rdf_predicate"]:
                raise ValueError("Evidence record endpoints/predicate disagree with their semantic relationship.")
            observations[values["relationship_id"]].append(Evidence(values["id"], values["semantic_uri"], values["evidence_state"],
                tuple(values["source"]), values.get("source_text"), values.get("weight")))
        entities = []
        for row in node_rows:
            values = dict(row["properties"])
            from src.kg.semantics import domain_labels
            labels = [label for label in row["labels"] if label in domain_labels()]
            if len(labels) != 1:
                raise ValueError("Managed entities must have exactly one established domain label.")
            label = labels[0]
            for key in ("managed_by", "dataset_id"):
                values.pop(key)
            identifier, name, uri = values.pop("id"), values.pop("name"), values.pop("semantic_uri")
            entities.append(Entity(label, identifier, name, uri, values))
        relationships = tuple(SemanticRelationship(row["properties"]["id"], row["source_uri"], row["target_uri"], row["relationship_type"],
                              tuple(observations[row["properties"]["id"]])) for row in relationship_rows)
        return GraphData(tuple(entities), relationships)

    def counts(self) -> dict:
        return {"nodes": self.connection.read(queries.COUNTS, dataset_id=self.dataset_id),
                "relationships": self.connection.read(queries.RELATIONSHIP_COUNTS, dataset_id=self.dataset_id),
                "evidence_states": self.connection.read(queries.EVIDENCE_COUNTS, dataset_id=self.dataset_id),
                "relationship_evidence_states": self.connection.read(queries.RELATIONSHIP_EVIDENCE_COUNTS, dataset_id=self.dataset_id)}

    def reset_development_graph(self, *, confirm: bool = False) -> None:
        """Delete only this loader-owned development dataset, never the database."""
        if not confirm or self.dataset_id not in {"synthetic_v1", "ontology_example"}:
            raise ValueError("Reset requires deliberate confirmation and a known development dataset.")
        self.connection.write("MATCH (n:FoodPreferenceKG {dataset_id: $dataset_id, managed_by: $owner}) DETACH DELETE n",
                              dataset_id=self.dataset_id, owner=MANAGED_BY)
