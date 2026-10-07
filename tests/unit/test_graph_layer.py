from dataclasses import replace
import unittest

from rdflib import Graph, Literal
from rdflib.namespace import RDFS

from src.kg.compatibility import ConstraintEngine
from src.kg.models import Entity
from src.kg.rdf_loader import extract_graph, load_rdf
from src.kg.semantics import ENTITY_PREFIX, FP, ROOT, domain_labels, relation_definitions
from tests.helpers import change_evidence, expected, rewire_relationship, seed_data
from src.kg.validation import validate_records


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = load_rdf(ROOT / "ontology/examples/minimum_domain.ttl")

    def test_existing_example_maps_exact_ids_iris_labels(self):
        lookup = {entity.id: entity for entity in self.example.entities}
        self.assertEqual(lookup["chickpea"].label, "Ingredient")
        self.assertEqual(lookup["chickpea"].semantic_uri, ENTITY_PREFIX + "chickpea")
        self.assertEqual(lookup["chickpea"].name, "Chickpea")
        self.assertEqual(lookup["dish_001"].properties["composition_complete"], True)
        self.assertEqual(lookup["dish_004"].properties["composition_complete"], False)
        self.assertEqual({edge.relationship_type for edge in self.example.relationships}, set(relation_definitions()))

    def test_missing_completeness_remains_unasserted(self):
        graph = Graph().parse(ROOT / "ontology/examples/minimum_domain.ttl")
        graph.remove((None, FP.compositionComplete, None))
        result = extract_graph(graph)
        self.assertTrue(all("composition_complete" not in entity.properties for entity in result.entities))

    def test_renamed_label_keeps_identity(self):
        graph = Graph().parse(ROOT / "ontology/examples/minimum_domain.ttl")
        subject = next(entity.semantic_uri for entity in self.example.entities if entity.id == "chickpea")
        from rdflib import URIRef
        graph.set((URIRef(subject), RDFS.label, Literal("Updated display name", lang="en")))
        entity = next(entity for entity in extract_graph(graph).entities if entity.id == "chickpea")
        self.assertEqual(entity.semantic_uri, subject)
        self.assertEqual(entity.name, "Updated display name")

    def test_invalid_weight_is_rejected_before_ingestion(self):
        graph = Graph().parse(ROOT / "ontology/examples/minimum_domain.ttl")
        from rdflib import URIRef
        graph.remove((URIRef(ENTITY_PREFIX + "statement_001"), FP.preferenceWeight, None))
        with self.assertRaisesRegex(ValueError, "SHACL validation"):
            extract_graph(graph)

    def test_untrusted_node_labels_cannot_become_cypher(self):
        with self.assertRaisesRegex(ValueError, "Unknown domain"):
            Entity("Dish) DELETE n //", "dish_001", "Dish", "urn:dish:001")

    def test_id_and_ownership_cannot_be_injected_through_properties(self):
        with self.assertRaisesRegex(ValueError, "cannot override"):
            Entity("Dish", "dish_001", "Dish", "urn:dish:001", {"dataset_id": "other"})


class ConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = seed_data()
        cls.engine = ConstraintEngine(cls.data)

    def test_hand_specified_status_fixtures(self):
        for case in expected()["cases"]:
            with self.subTest(case=case):
                result = self.engine.evaluate(case["profile_id"], case["dish_id"])
                self.assertEqual(result.status, case["status"])

    def test_seed_sizes_and_source_text(self):
        for label, count in (("UserProfile", 8), ("Restaurant", 5), ("Dish", 20), ("Ingredient", 26)):
            self.assertEqual(sum(entity.label == label for entity in self.data.entities), count)
        edge = next(edge for edge in self.data.relationships if self.engine.entities[edge.source_uri].id == "seed_dish_002" and edge.relationship_type == "CONTAINS" and self.engine.entities[edge.target_uri].id == "ingredient_cashew")
        self.assertEqual(len(edge.observations), 2)
        self.assertEqual({record.evidence_state for record in edge.observations}, {"CONFIRMED", "INFERRED"})
        self.assertTrue(all(record.source_text for record in edge.observations))

    def test_confirmed_conflict_survives_multiple_uncertain_observations(self):
        result = self.engine.evaluate("seed_profile_005", "seed_dish_002")
        self.assertEqual(result.status, "INCOMPATIBLE")
        reason = next(reason for reason in result.reasons if reason["outcome"] == "INCOMPATIBLE")
        self.assertEqual(reason["ingredient_id"], "ingredient_cashew")
        self.assertEqual(reason["allergen_id"], "allergen_tree_nut")
        self.assertEqual([step["relationship_type"] for step in reason["path"]], ["ALLERGIC_TO", "CONTAINS", "HAS_ALLERGEN"])

    def test_inferred_or_possible_allergy_paths_are_unknown(self):
        for relation_source, relationship_type, target in (("seed_dish_002", "CONTAINS", "ingredient_cashew"), ("ingredient_cashew", "HAS_ALLERGEN", "allergen_tree_nut")):
            for state in ("INFERRED", "MAY_CONTAIN", "UNKNOWN"):
                with self.subTest(type=relationship_type, state=state):
                    changed = change_evidence(self.data, relation_source, relationship_type, target, state)
                    self.assertEqual(ConstraintEngine(changed).evaluate("seed_profile_005", "seed_dish_002").status, "UNKNOWN")

    def test_absent_allergen_edges_and_complete_composition_do_not_prove_clearance(self):
        self.assertEqual(self.engine.entity("Dish", "seed_dish_001").properties["composition_complete"], True)
        self.assertEqual(self.engine.evaluate("seed_profile_005", "seed_dish_001").status, "UNKNOWN")

    def test_soft_vegetarian_diet_does_not_reject_beef_with_confirmed_required_diet(self):
        self.assertEqual(self.engine.evaluate("seed_profile_002", "seed_dish_003").status, "COMPATIBLE")

    def test_preferred_health_goal_is_not_mandatory(self):
        self.assertEqual(self.engine.evaluate("seed_profile_004", "seed_dish_003").status, "COMPATIBLE")
        self.assertEqual(self.engine.evaluate("seed_profile_008", "seed_dish_003").status, "INCOMPATIBLE")

    def test_diet_allergen_exclusion_activates_the_actual_allergy_path(self):
        changed = rewire_relationship(self.data, "seed_restriction_002", "EXCLUDES_ALLERGEN", "allergen_peanut", new_source_id="diet_gluten_free")
        validate_records(changed)
        engine = ConstraintEngine(changed)
        result = engine.evaluate("seed_profile_002", "seed_dish_009")
        reason = next(reason for reason in result.reasons if reason["constraint_type"] == "ALLERGY" and reason["outcome"] == "INCOMPATIBLE")
        self.assertEqual([step["relationship_type"] for step in reason["path"]], ["REQUIRES_DIET", "EXCLUDES_ALLERGEN", "CONTAINS", "HAS_ALLERGEN"])
        self.assertEqual(engine.evaluate("seed_profile_002", "seed_dish_011").status, "UNKNOWN")

    def test_soft_allergen_exclusion_goal_returns_penalty_evidence_without_rejection(self):
        changed = rewire_relationship(self.data, "seed_profile_004", "PREFERS_HEALTH_GOAL", "seed_goal_001", new_target_id="seed_goal_002")
        validate_records(changed)
        engine = ConstraintEngine(changed)
        self.assertEqual(engine.evaluate("seed_profile_004", "seed_dish_002").status, "COMPATIBLE")
        match = next(row for row in engine.soft_matches("seed_profile_004", "seed_dish_002") if row["preference_type"] == "PREFERS_HEALTH_GOAL")
        self.assertEqual(match["effect"], "penalty")
        self.assertEqual([step["relationship_type"] for step in match["path"]], ["PREFERS_HEALTH_GOAL", "EXCLUDES_ALLERGEN", "CONTAINS", "HAS_ALLERGEN"])

    def test_zero_soft_weight_does_not_disable_avoidance(self):
        preferences = self.engine.soft_preferences("seed_profile_006")
        self.assertEqual(preferences[0]["weights"], [0.0])
        self.assertEqual(self.engine.evaluate("seed_profile_006", "seed_dish_006").status, "INCOMPATIBLE")

    def test_uncertain_cuisine_does_not_change_mandatory_diet_decision(self):
        changed = change_evidence(self.data, "seed_dish_001", "HAS_CUISINE", "cuisine_indian", "UNKNOWN")
        self.assertEqual(ConstraintEngine(changed).evaluate("seed_profile_002", "seed_dish_001").status, "COMPATIBLE")

    def test_missing_required_diet_assessment_is_unknown(self):
        dish = self.engine.entity("Dish", "seed_dish_001")
        changed = replace(self.data, relationships=tuple(edge for edge in self.data.relationships if not (
            edge.source_uri == dish.semantic_uri and edge.relationship_type == "COMPATIBLE_WITH_DIET" and self.engine.entities[edge.target_uri].id == "diet_gluten_free")))
        self.assertEqual(ConstraintEngine(changed).evaluate("seed_profile_002", "seed_dish_001").status, "UNKNOWN")

    def test_unconfirmed_mandatory_profile_link_cannot_establish_confirmed_conflict(self):
        changed = change_evidence(self.data, "seed_profile_006", "MUST_AVOID", "ingredient_mushroom", "UNKNOWN")
        self.assertEqual(ConstraintEngine(changed).evaluate("seed_profile_006", "seed_dish_006").status, "UNKNOWN")

    def test_only_compatible_candidates_receive_soft_match_evidence(self):
        groups = self.engine.candidate_groups("seed_profile_002")
        expected_values = expected()
        for status, key in (("COMPATIBLE", "profile2_compatible"), ("INCOMPATIBLE", "profile2_incompatible"), ("UNKNOWN", "profile2_unknown")):
            self.assertEqual({row["dish_id"] for row in groups[status]}, set(expected_values[key]))
        self.assertTrue(all(not row["soft_matches"] for status in ("UNKNOWN", "INCOMPATIBLE") for row in groups[status]))

    def test_unknown_profile_or_dish_is_an_error_not_a_compatibility_claim(self):
        with self.assertRaisesRegex(ValueError, "Unknown UserProfile"):
            self.engine.evaluate("missing", "seed_dish_001")
        with self.assertRaisesRegex(ValueError, "Unknown Dish"):
            self.engine.evaluate("seed_profile_002", "missing")


if __name__ == "__main__":
    unittest.main()
