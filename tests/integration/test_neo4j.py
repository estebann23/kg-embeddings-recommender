"""Read-only checks for the isolated HUMMUS recipe pilot."""
import unittest

from src.kg.connection import Neo4jConnection


class HummusPilotGraphTests(unittest.TestCase):
    def test_recipe_path_and_training_only_ratings(self):
        try:
            connection = Neo4jConnection()
            connection.verify()
        except (ConnectionError, ValueError) as error:
            self.skipTest(f"Neo4j development instance unavailable: {error}")
        try:
            rows = connection.read(
                "MATCH (u:ObservedUser:HummusPilot {dataset_id: $dataset_id})-"
                "[rating:RATED]->(recipe:Recipe:HummusPilot)-"
                "[:CONTAINS_MAPPED_INGREDIENT]->(ingredient:Ingredient:HummusPilot)-"
                "[:MAPS_TO_FOODON]->(concept:FoodOnConcept:HummusPilot) "
                "RETURN u.id AS user_id, recipe.id AS recipe_id, "
                "ingredient.external_iri AS ingredient_iri, "
                "concept.external_iri AS foodon_iri LIMIT 1",
                dataset_id="hummus_pilot_v1",
            )
            self.assertTrue(rows)
            counts = connection.read(
                "MATCH (:ObservedUser:HummusPilot {dataset_id: $dataset_id})-"
                "[r:RATED]->(:Recipe:HummusPilot) RETURN count(r) AS count",
                dataset_id="hummus_pilot_v1",
            )
            self.assertEqual(counts[0]["count"], 240)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
