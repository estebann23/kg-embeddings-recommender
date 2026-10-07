"""Offline vocabulary and SHACL checks; no database writes or scoring logic."""
from pathlib import Path
import unittest

from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate
from rdflib import Graph, Literal, Namespace
from rdflib.collection import Collection
from rdflib.compare import isomorphic
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, XSD


ROOT = Path(__file__).resolve().parents[1]
FP = Namespace("https://github.com/estebann23/kg-embeddings-recommender/ontology/food_preferences#")
EX = Namespace("https://github.com/estebann23/kg-embeddings-recommender/entity/")
DOMAIN_CLASSES = {
    "UserProfile", "Restaurant", "Dish", "Ingredient", "Cuisine", "Taste",
    "DietaryPattern", "Allergen", "HealthRestriction", "HealthGoal",
}
HARD = {
    "requiresDiet", "allergicTo", "mustAvoid", "requiresHealthRestriction",
    "requiresHealthGoal",
}
SOFT = {
    "followsDiet", "likesIngredient", "dislikesIngredient", "prefersCuisine",
    "prefersTaste", "prefersHealthGoal",
}
RELATIONSHIPS = {
    "followsDiet": ("UserProfile", "DietaryPattern", "FOLLOWS_DIET"),
    "requiresDiet": ("UserProfile", "DietaryPattern", "REQUIRES_DIET"),
    "allergicTo": ("UserProfile", "Allergen", "ALLERGIC_TO"),
    "mustAvoid": ("UserProfile", "Ingredient", "MUST_AVOID"),
    "likesIngredient": ("UserProfile", "Ingredient", "LIKES_INGREDIENT"),
    "dislikesIngredient": ("UserProfile", "Ingredient", "DISLIKES_INGREDIENT"),
    "prefersCuisine": ("UserProfile", "Cuisine", "PREFERS_CUISINE"),
    "prefersTaste": ("UserProfile", "Taste", "PREFERS_TASTE"),
    "requiresHealthRestriction": ("UserProfile", "HealthRestriction", "REQUIRES_HEALTH_RESTRICTION"),
    "prefersHealthGoal": ("UserProfile", "HealthGoal", "PREFERS_HEALTH_GOAL"),
    "requiresHealthGoal": ("UserProfile", "HealthGoal", "REQUIRES_HEALTH_GOAL"),
    "offers": ("Restaurant", "Dish", "OFFERS"),
    "contains": ("Dish", "Ingredient", "CONTAINS"),
    "hasCuisine": ("Dish", "Cuisine", "HAS_CUISINE"),
    "hasTaste": ("Dish", "Taste", "HAS_TASTE"),
    "hasAllergen": ("Ingredient", "Allergen", "HAS_ALLERGEN"),
    "excludesIngredient": (None, "Ingredient", "EXCLUDES_INGREDIENT"),
    "excludesAllergen": (None, "Allergen", "EXCLUDES_ALLERGEN"),
    "compatibleWithDiet": ("Dish", "DietaryPattern", "COMPATIBLE_WITH_DIET"),
    "incompatibleWithDiet": ("Dish", "DietaryPattern", "INCOMPATIBLE_WITH_DIET"),
}


class OntologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Graph().parse(ROOT / "ontology/food_preferences.ttl")
        cls.shapes = Graph().parse(ROOT / "ontology/shapes/food_preferences.shacl.ttl")
        cls.example = Graph().parse(ROOT / "ontology/examples/minimum_domain.ttl")

    def example_copy(self):
        graph = Graph()
        graph += self.example
        return graph

    def check_graph(self, graph, expected=True, message=None):
        conforms, _, report = validate(
            graph, shacl_graph=self.shapes, ont_graph=self.vocabulary,
            inference="none", advanced=False,
        )
        self.assertEqual(conforms, expected, report)
        if message:
            self.assertIn(message, report)

    def test_exactly_ten_named_domain_classes(self):
        classes = {
            str(node).removeprefix(str(FP))
            for node in self.vocabulary.subjects(RDF.type, OWL.Class)
            if str(node).startswith(str(FP))
        }
        self.assertEqual(classes, DOMAIN_CLASSES)
        for name in DOMAIN_CLASSES:
            self.assertIsNotNone(self.vocabulary.value(FP[name], RDFS.label))
            self.assertIsNotNone(self.vocabulary.value(FP[name], RDFS.comment))

    def test_relationship_directions_and_documented_mapping(self):
        properties = set(self.vocabulary.subjects(RDF.type, OWL.ObjectProperty))
        self.assertEqual(properties, {FP[name] for name in RELATIONSHIPS})
        documentation = (ROOT / "ontology/DOMAIN_MODEL.txt").read_text(encoding="utf-8")
        for name, (domain, target, mapping) in RELATIONSHIPS.items():
            with self.subTest(property=name):
                self.assertEqual(self.vocabulary.value(FP[name], RDFS.range), FP[target])
                if domain:
                    self.assertEqual(self.vocabulary.value(FP[name], RDFS.domain), FP[domain])
                else:
                    union = self.vocabulary.value(FP[name], RDFS.domain)
                    members = Collection(self.vocabulary, self.vocabulary.value(union, OWL.unionOf))
                    self.assertEqual(set(members), {FP.DietaryPattern, FP.HealthRestriction, FP.HealthGoal})
                self.assertIn(f"{name} |", documentation)
                self.assertIn(f"| {mapping} |", documentation)
                self.assertIsNotNone(self.vocabulary.value(FP[name], RDFS.comment))

    def test_hard_and_soft_are_explicit_and_disjoint(self):
        for name in HARD:
            self.assertEqual(self.vocabulary.value(FP[name], FP.constraintKind), Literal("hard"))
        for name in SOFT:
            self.assertEqual(self.vocabulary.value(FP[name], FP.constraintKind), Literal("soft"))
        self.assertFalse(HARD & SOFT)

    def test_turtle_round_trip_preserves_graphs(self):
        for graph in (self.vocabulary, self.shapes, self.example):
            self.assertTrue(isomorphic(graph, Graph().parse(data=graph.serialize(format="turtle"), format="turtle")))

    def test_valid_example(self):
        self.check_graph(self.example_copy())

    def test_missing_or_blank_names_fail(self):
        for value in (None, Literal("   ")):
            with self.subTest(value=value):
                graph = self.example_copy()
                graph.remove((EX.profile_001, RDFS.label, None))
                if value is not None:
                    graph.add((EX.profile_001, RDFS.label, value))
                self.check_graph(graph, False, "nonblank")

    def test_orphan_dish_fails(self):
        graph = self.example_copy()
        graph.remove((EX.restaurant_001, FP.offers, EX.dish_001))
        self.check_graph(graph, False, "offered by at least one")

    def test_wrong_relationship_target_fails_without_inference(self):
        graph = self.example_copy()
        graph.remove((EX.indian, RDF.type, FP.Cuisine))
        graph.add((EX.indian, RDF.type, FP.Ingredient))
        self.check_graph(graph, False, "prefersCuisine must point to a typed Cuisine")

    def test_wrong_relationship_subject_fails(self):
        graph = self.example_copy()
        graph.remove((EX.profile_001, RDF.type, FP.UserProfile))
        graph.add((EX.profile_001, RDF.type, FP.Restaurant))
        self.check_graph(graph, False)

    def test_invalid_weight_values_fail(self):
        for value in (Literal("-0.1", datatype=XSD.decimal), Literal("1.1", datatype=XSD.decimal), Literal("0.5")):
            with self.subTest(value=value):
                graph = self.example_copy()
                graph.set((EX.statement_001, FP.preferenceWeight, value))
                self.check_graph(graph, False, "Preference weights")

    def test_missing_soft_weight_fails(self):
        graph = self.example_copy()
        graph.remove((EX.statement_001, FP.preferenceWeight, None))
        self.check_graph(graph, False, "soft preference statement requires a weight")

    def test_hard_constraint_weight_fails(self):
        graph = self.example_copy()
        graph.add((EX.statement_002, FP.preferenceWeight, Literal("1.0", datatype=XSD.decimal)))
        self.check_graph(graph, False, "never on hard restrictions")

    def test_invalid_or_missing_evidence_fails(self):
        for value in (None, Literal("SAFE")):
            with self.subTest(value=value):
                graph = self.example_copy()
                graph.remove((EX.statement_016, FP.evidenceState, None))
                if value is not None:
                    graph.add((EX.statement_016, FP.evidenceState, value))
                self.check_graph(graph, False, "Statement evidence")

    def test_missing_source_fails(self):
        graph = self.example_copy()
        graph.remove((EX.statement_016, DCTERMS.source, None))
        self.check_graph(graph, False, "source identifier")

    def test_missing_relationship_metadata_fails(self):
        graph = self.example_copy()
        graph.remove((EX.statement_016, None, None))
        self.check_graph(graph, False, "Every relationship needs")

    def test_detached_statement_fails(self):
        graph = self.example_copy()
        graph.remove((EX.dish_001, FP.contains, EX.chickpea))
        self.check_graph(graph, False, "existing direct relationship")

    def test_multiple_source_observations_are_preserved(self):
        graph = self.example_copy()
        for predicate, value in list(graph.predicate_objects(EX.statement_016)):
            graph.add((EX.duplicate_statement, predicate, value))
        graph.set((EX.duplicate_statement, DCTERMS.source, Literal("Second synthetic source")))
        graph.set((EX.duplicate_statement, FP.evidenceState, Literal("INFERRED")))
        self.check_graph(graph)
        self.assertEqual(graph.value(EX.statement_016, FP.evidenceState), Literal("CONFIRMED"))
        self.assertEqual(graph.value(EX.duplicate_statement, FP.evidenceState), Literal("INFERRED"))

    def test_contradictory_dietary_assessments_fail(self):
        graph = self.example_copy()
        graph.add((EX.dish_001, FP.incompatibleWithDiet, EX.vegan))
        self.check_graph(graph, False, "both positive and negative")

    def test_uncertainty_states_are_supported_but_not_clearance(self):
        for value in ("INFERRED", "MAY_CONTAIN", "UNKNOWN"):
            with self.subTest(value=value):
                graph = self.example_copy()
                graph.set((EX.statement_019, FP.evidenceState, Literal(value)))
                self.check_graph(graph)
                self.assertNotIn((EX.dish_004, FP.compatibleWithDiet, EX.vegan), graph)

    def test_may_contain_is_not_a_dietary_clearance(self):
        graph = self.example_copy()
        graph.set((EX.statement_027, FP.evidenceState, Literal("MAY_CONTAIN")))
        self.check_graph(graph, False, "reserved for possible ingredient or allergen")

    def test_nonboolean_composition_fails(self):
        graph = self.example_copy()
        graph.set((EX.dish_004, FP.compositionComplete, Literal("yes")))
        self.check_graph(graph, False, "one boolean value")

    def test_preference_and_allergy_conflict_is_representable(self):
        query = """
            PREFIX fp: <https://github.com/estebann23/kg-embeddings-recommender/ontology/food_preferences#>
            SELECT ?ingredient ?allergen WHERE {
                ?profile fp:likesIngredient ?ingredient ; fp:allergicTo ?allergen .
                ?ingredient fp:hasAllergen ?allergen .
            }
        """
        self.assertEqual(list(self.example.query(query)), [(EX.almond, EX.tree_nut)])
        self.check_graph(self.example_copy())

    def test_owl_does_not_promote_soft_goals_or_conflate_rule_classes(self):
        graph = self.example_copy()
        graph += self.vocabulary
        DeductiveClosure(OWLRL_Semantics).expand(graph)
        self.assertNotIn((EX.profile_001, FP.requiresHealthGoal, EX.goal_001), graph)
        self.assertNotIn((EX.restriction_001, RDF.type, FP.DietaryPattern), graph)
        self.assertNotIn((EX.goal_001, RDF.type, FP.HealthRestriction), graph)
        self.assertNotIn((EX.dish_004, FP.compatibleWithDiet, EX.vegan), graph)


if __name__ == "__main__":
    unittest.main()
