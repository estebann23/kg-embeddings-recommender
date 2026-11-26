"""Recipe graph and scenario constraint checks."""
import unittest

from src.kg.compatibility import ConstraintEngine
from src.kg.models import Entity
from src.kg.semantics import FP, ROOT
from src.kg.rdf_loader import load_rdf
from src.kg.validation import validate_records


class RecipeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_rdf(ROOT / "data/synthetic/seed.ttl")
        cls.engine = ConstraintEngine(cls.data)

    def test_recipe_paths_and_external_foodon_ids(self):
        recipe = self.engine.entity("Recipe", "recipe_1")
        ingredient = self.engine.edges(recipe.semantic_uri, "CONTAINS")[0]
        foodon = self.engine.edges(ingredient.target_uri, "MAPS_TO_FOODON")[0]
        self.assertTrue(foodon.target_uri.startswith("https://example.org/fixture/foodon/"))
        self.assertEqual(self.engine.entities[foodon.target_uri].label, "FoodOnConcept")
        self.assertFalse(any(entity.label in {"Restaurant", "Dish"} for entity in self.data.entities))

    def test_scenario_detects_confirmed_conflict_and_unknown(self):
        self.assertEqual(self.engine.evaluate("scenario_1", "recipe_1").status, "INCOMPATIBLE")
        self.assertEqual(self.engine.evaluate("scenario_1", "recipe_2").status, "UNKNOWN")
        groups = self.engine.candidate_groups("scenario_1")
        self.assertEqual({row["recipe_id"] for row in groups["INCOMPATIBLE"]}, {"recipe_1"})
        self.assertEqual({row["recipe_id"] for row in groups["UNKNOWN"]}, {"recipe_2"})

    def test_scenario_not_observed_user(self):
        with self.assertRaisesRegex(ValueError, "Unknown ScenarioProfile"):
            self.engine.evaluate("user_1", "recipe_1")

    def test_validation_round_trip(self):
        validate_records(self.data)

    def test_deprecated_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown domain label"):
            Entity("Dish", "old_item", "Old item", "urn:old:item")


if __name__ == "__main__":
    unittest.main()
