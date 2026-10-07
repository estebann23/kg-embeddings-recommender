"""Opt-in tests against only the loader-owned synthetic development dataset."""
from dataclasses import replace
import os
import unittest
from uuid import uuid4

from neo4j.exceptions import ConstraintError

from src.kg.compatibility import ConstraintEngine
from src.kg.competency import run_competency_queries
from src.kg.connection import Neo4jConnection
from src.kg.repository import GraphRepository
from src.kg.schema import initialize_schema
from src.kg.semantics import ENTITY_PREFIX, domain_labels, relation_definitions
from src.kg.validation import validate_operational_graph
from tests.helpers import expected, seed_data


@unittest.skipUnless(os.environ.get("NEO4J_INTEGRATION") == "1", "Set NEO4J_INTEGRATION=1 for development database tests.")
class Neo4jTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connection = Neo4jConnection()
        cls.addClassCleanup(cls.connection.close)
        cls.connection.verify()
        initialize_schema(cls.connection)
        cls.repository = GraphRepository(cls.connection)
        cls.source = seed_data()
        cls.repository.load(cls.source)

    def test_connection_and_schema(self):
        self.assertEqual(self.connection.read("RETURN 1 AS ok"), [{"ok": 1}])
        constraints = {row["name"] for row in self.connection.read("SHOW CONSTRAINTS YIELD name RETURN name")}
        self.assertTrue({"fp_" + label.lower() + "_id" for label in domain_labels()} <= constraints)
        initialize_schema(self.connection)
        after = {row["name"] for row in self.connection.read("SHOW CONSTRAINTS YIELD name RETURN name")}
        self.assertEqual(constraints, after)

    def test_unique_id_constraint_rejects_duplicate_in_rolled_back_transaction(self):
        def duplicate(tx):
            tx.run("CREATE (:Dish {id: $id})", id="seed_dish_001").consume()
            raise AssertionError("A duplicate ID was accepted.")
        with self.assertRaises(ConstraintError):
            self.connection.write_transaction(duplicate)

    def test_full_mapping_metadata_and_multiple_evidence_preserved(self):
        actual = validate_operational_graph(self.repository)
        entities = {entity.semantic_uri: entity for entity in actual.entities}
        for entity in self.source.entities:
            self.assertEqual(entities[entity.semantic_uri], entity)
        edges = {edge.id: edge for edge in actual.relationships}
        for edge in self.source.relationships:
            self.assertEqual(edges[edge.id], edge)
        self.assertEqual(len(entities), 83)
        self.assertEqual(sum(len(edge.observations) for edge in actual.relationships), 237)
        self.assertEqual({edge.relationship_type for edge in actual.relationships}, set(relation_definitions()))
        summary = self.connection.read(
            "MATCH (d:Dish {id: $dish})-[r:CONTAINS]->(i:Ingredient {id: $ingredient}) RETURN properties(r) AS metadata",
            dish="seed_dish_002", ingredient="ingredient_cashew",
        )[0]["metadata"]
        self.assertTrue(summary["evidence_conflict"])
        self.assertEqual(summary["evidence_state"], "CONFIRMED")
        self.assertEqual(len(summary["evidence_uris"]), 2)
        self.assertEqual(len(summary["source"]), 2)

    def test_duplicate_load_is_idempotent(self):
        before = self.repository.counts()
        self.repository.load(self.source)
        self.assertEqual(self.repository.counts(), before)

    def test_cross_dataset_identity_collision_rolls_back(self):
        before = self.repository.counts()
        with self.assertRaisesRegex(ValueError, "different import"):
            GraphRepository(self.connection, "other_fixture").load(self.source)
        self.assertEqual(self.repository.counts(), before)
        self.assertEqual(self.connection.read("MATCH (n:FoodPreferenceKG {dataset_id:'other_fixture'}) RETURN count(n) AS count"), [{"count": 0}])

    def test_node_label_update_keeps_id_and_semantic_uri(self):
        changed = replace(self.source, entities=tuple(replace(entity, name="Temporary renamed dish",
                           properties={**entity.properties, "label_values": ["Temporary renamed dish"]})
                           if entity.id == "seed_dish_001" else entity for entity in self.source.entities))
        try:
            self.repository.load(changed)
            row = self.connection.read("MATCH (d:Dish {id:$id}) RETURN d.id AS id, d.semantic_uri AS uri, d.name AS name", id="seed_dish_001")[0]
            self.assertEqual(row, {"id": "seed_dish_001", "uri": ENTITY_PREFIX + "seed_dish_001", "name": "Temporary renamed dish"})
        finally:
            self.repository.load(self.source)

    def test_scoped_reset_and_rebuild_preserve_unrelated_data(self):
        marker = str(uuid4())
        self.connection.write("CREATE (:GraphLayerTestSentinel {id: $id})", id=marker)
        before = self.repository.graph_data()
        try:
            with self.assertRaises(ValueError):
                self.repository.reset_development_graph()
            self.repository.reset_development_graph(confirm=True)
            self.assertEqual(self.repository.graph_data().entities, ())
            self.assertEqual(self.connection.read("MATCH (n:GraphLayerTestSentinel {id:$id}) RETURN count(n) AS count", id=marker), [{"count": 1}])
            self.repository.load(self.source)
            self.assertEqual(self.repository.graph_data(), before)
        finally:
            self.connection.write("MATCH (n:GraphLayerTestSentinel {id:$id}) DELETE n", id=marker)

    def test_removed_completeness_assertion_is_not_kept_as_stale_true(self):
        changed = replace(self.source, entities=tuple(replace(entity, properties={key: value for key, value in entity.properties.items() if key != "composition_complete"})
                           if entity.id == "seed_dish_001" else entity for entity in self.source.entities))
        try:
            self.repository.load(changed)
            engine = ConstraintEngine(self.repository.graph_data())
            self.assertNotIn("composition_complete", engine.entity("Dish", "seed_dish_001").properties)
            self.assertEqual(engine.evaluate("seed_profile_002", "seed_dish_001").status, "UNKNOWN")
        finally:
            self.repository.load(self.source)

    def test_misattached_evidence_is_reported_instead_of_fabricating_a_path(self):
        edge = next(edge for edge in self.source.relationships if edge.relationship_type == "HAS_ALLERGEN")
        identifier = edge.observations[0].id
        try:
            self.connection.write("MATCH (e:EvidenceRecord {id:$id}) SET e.source_uri = $uri", id=identifier, uri=ENTITY_PREFIX + "seed_dish_001")
            with self.assertRaisesRegex(ValueError, "endpoints/predicate disagree"):
                self.repository.graph_data()
        finally:
            self.repository.load(self.source)


@unittest.skipUnless(os.environ.get("NEO4J_INTEGRATION") == "1", "Set NEO4J_INTEGRATION=1 for development database tests.")
class CompetencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connection = Neo4jConnection()
        cls.addClassCleanup(cls.connection.close)
        cls.connection.verify()
        cls.repository = GraphRepository(cls.connection)
        if not cls.repository.graph_data().entities:
            initialize_schema(cls.connection)
            cls.repository.load(seed_data())
        cls.engine = ConstraintEngine(validate_operational_graph(cls.repository))
        cls.expected = expected()
        cls.queries = {profile: run_competency_queries(cls.engine, profile)
                       for profile in (f"seed_profile_{i:03d}" for i in range(1, 9))}

    def test_cq1_confirmed_dietary_compatibility(self):
        self.assertEqual(set(self.queries["seed_profile_002"]["CQ1"]), set(self.expected["cq1_vegan"]))

    def test_cq2_confirmed_allergy_paths(self):
        self.assertEqual({row["dish_id"] for row in self.queries["seed_profile_003"]["CQ2"]}, {"seed_dish_009"})
        self.assertEqual({row["dish_id"] for row in self.queries["seed_profile_005"]["CQ2"]}, {"seed_dish_002"})

    def test_cq3_explicit_ingredient_avoidance(self):
        incompatible = {row["dish_id"] for row in self.queries["seed_profile_006"]["CQ3"] if row["outcome"] == "INCOMPATIBLE"}
        self.assertEqual(incompatible, {"seed_dish_006"})

    def test_cq4_mandatory_diet_activates_exclusion_rules(self):
        incompatible = {row["dish_id"] for row in self.queries["seed_profile_002"]["CQ4"] if row["outcome"] == "INCOMPATIBLE"}
        self.assertEqual(incompatible, set(self.expected["profile2_incompatible"]))

    def test_cq5_restaurants_need_compatible_dishes(self):
        self.assertEqual(self.queries["seed_profile_002"]["CQ5"], self.expected["profile2_restaurants"])
        self.assertEqual(self.queries["seed_profile_001"]["CQ5"], [])

    def test_cq6_all_six_soft_relationships_preserve_weights(self):
        preferences = self.queries["seed_profile_004"]["CQ6"]
        self.assertEqual({row["relationship_type"] for row in preferences}, {"FOLLOWS_DIET", "LIKES_INGREDIENT", "DISLIKES_INGREDIENT", "PREFERS_CUISINE", "PREFERS_TASTE", "PREFERS_HEALTH_GOAL"})
        self.assertTrue(all(0 <= weight <= 1 for row in preferences for weight in row["weights"]))

    def test_cq7_hard_constraints_override_positive_and_zero_weight_preferences(self):
        self.assertEqual({row["dish_id"] for row in self.queries["seed_profile_005"]["CQ7"]}, {"seed_dish_002"})
        self.assertEqual({row["dish_id"] for row in self.queries["seed_profile_006"]["CQ7"]}, {"seed_dish_006"})

    def test_cq8_reasons_preserve_original_observations(self):
        cases = self.queries["seed_profile_005"]["CQ8"]
        reason = next(reason for case in cases for reason in case["reasons"] if reason["outcome"] == "INCOMPATIBLE")
        self.assertEqual(reason["ingredient_id"], "ingredient_cashew")
        for step in reason["path"]:
            self.assertTrue(step["relationship_id"])
            self.assertTrue(all(record["semantic_uri"] and record["source"] for record in step["observations"]))

    def test_cq9_uncertainty_never_enters_compatible_set(self):
        self.assertEqual({row["dish_id"] for row in self.queries["seed_profile_002"]["CQ9"]}, set(self.expected["profile2_unknown"]))

    def test_cq10_positive_match_paths_are_real_graph_paths(self):
        evidence = self.queries["seed_profile_004"]["CQ10"]
        sample = next(row for row in evidence if row["dish_id"] == "seed_dish_013" and row["preference_type"] == "PREFERS_TASTE")
        self.assertEqual([step["relationship_type"] for step in sample["path"]], ["PREFERS_TASTE", "HAS_TASTE", "OFFERS"])
        self.assertEqual(sample["value_id"], "taste_savory")
        self.assertEqual(sample["restaurant_id"], "seed_restaurant_004")
        graph = {edge.id: edge for edge in self.engine.data.relationships}
        for row in evidence:
            for step in row["path"]:
                self.assertIn(step["relationship_id"], graph)


if __name__ == "__main__":
    unittest.main()
