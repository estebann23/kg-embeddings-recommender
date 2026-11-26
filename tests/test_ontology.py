"""Recipe-level RDF vocabulary and SHACL regression tests."""
import unittest

from pyshacl import validate
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from src.kg.rdf_loader import extract_graph
from src.kg.semantics import FP, ROOT, domain_labels, relation_definitions
from src.kg.validation import to_rdf


EX = Namespace("https://github.com/estebann23/kg-embeddings-recommender/entity/")


class RecipeOntologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab = Graph().parse(ROOT / "ontology/food_preferences.ttl")
        cls.example = Graph().parse(ROOT / "ontology/examples/minimum_domain.ttl")
        cls.shapes = ROOT / "ontology/shapes/food_preferences.shacl.ttl"

    def copy(self):
        graph = Graph()
        graph += self.example
        return graph

    def check(self, graph, expected=True):
        conforms, _, report = validate(graph, shacl_graph=str(self.shapes),
                                       ont_graph=self.vocab, inference="none")
        self.assertEqual(conforms, expected, str(report))

    def test_only_recipe_domain_classes(self):
        self.assertEqual(set(domain_labels()), {
            "ObservedUser", "ScenarioProfile", "Recipe", "Ingredient",
            "FoodOnConcept", "DietaryPattern", "Allergen",
        })
        self.assertNotIn(FP.Restaurant, set(self.vocab.subjects(RDF.type, OWL.Class)))
        self.assertNotIn(FP.Dish, set(self.vocab.subjects(RDF.type, OWL.Class)))
        self.assertNotIn(FP.offers, set(self.vocab.subjects(RDF.type, OWL.ObjectProperty)))

    def test_mapping_matches_vocab(self):
        definitions = relation_definitions()
        self.assertEqual(set(definitions), {
            "RATED", "SCENARIO_FOR", "CONTAINS", "MAPS_TO_FOODON",
            "FOODON_SUBCLASS_OF", "REQUIRES_DIET", "ALLERGIC_TO", "MUST_AVOID",
            "HAS_ALLERGEN", "EXCLUDES_INGREDIENT", "EXCLUDES_ALLERGEN",
            "COMPATIBLE_WITH_DIET", "INCOMPATIBLE_WITH_DIET",
        })
        self.assertEqual(definitions["RATED"].domains, frozenset({"ObservedUser"}))
        self.assertEqual(definitions["RATED"].target, "Recipe")

    def test_valid_example_and_round_trip(self):
        self.check(self.copy())
        data = extract_graph(self.copy())
        self.assertEqual(len(data.entities), 10)
        self.assertEqual(len(data.relationships), 10)
        self.assertEqual({record.rating for edge in data.relationships for record in edge.observations
                          if edge.relationship_type == "RATED"}, {5})
        self.check(to_rdf(data))

    def test_observed_user_cannot_receive_scenario_allergy(self):
        graph = self.copy()
        graph.add((EX.user_1, FP.allergicTo, EX.tree_nut))
        self.check(graph, False)

    def test_rated_statement_requires_1_to_5(self):
        for value in (None, Literal(0, datatype=XSD.integer), Literal(6, datatype=XSD.integer)):
            graph = self.copy()
            graph.remove((EX.recipe_statement_001, FP.ratingValue, None))
            if value is not None:
                graph.add((EX.recipe_statement_001, FP.ratingValue, value))
            self.check(graph, False)

    def test_relationship_needs_evidence(self):
        graph = self.copy()
        graph.remove((EX.recipe_statement_004, None, None))
        self.check(graph, False)

    def test_negative_nutrition_fails(self):
        graph = self.copy()
        graph.set((EX.recipe_1, FP.calories, Literal("-1", datatype=XSD.decimal)))
        self.check(graph, False)


if __name__ == "__main__":
    unittest.main()
