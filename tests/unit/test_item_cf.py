"""Boundary cases for the frozen rating-aware item-CF baseline."""

import math
import unittest

from src.recommenders.item_cf import ndcg_at_10, rank_for_n, score_candidates


class ItemCFTests(unittest.TestCase):
    def test_low_rating_is_negative_signal_but_still_a_defined_score(self):
        history = [("hummus_recipe_1", 1), ("hummus_recipe_2", 5)]
        adjacency = {
            "hummus_recipe_1": [("hummus_recipe_3", 0.8)],
            "hummus_recipe_2": [("hummus_recipe_3", 0.2)],
        }
        contributions = score_candidates(history, adjacency)
        fallback = ["hummus_recipe_4", "hummus_recipe_3", "hummus_recipe_1"]
        ranked, scorable = rank_for_n(history, contributions, 5, fallback, k=2)
        self.assertEqual(scorable, 1)
        self.assertEqual([row[0] for row in ranked], ["hummus_recipe_3", "hummus_recipe_4"])
        self.assertEqual([row[2] for row in ranked], ["score", "fallback"])
        self.assertTrue(math.isclose(ranked[0][1], -0.6))
        self.assertEqual(ndcg_at_10(ranked, {"hummus_recipe_3": 5}), 1.0)

    def test_ties_choose_numeric_recipe_id(self):
        history = [("hummus_recipe_1", 5)]
        adjacency = {"hummus_recipe_1": [("hummus_recipe_10", 0.5),
                                          ("hummus_recipe_2", 0.5)]}
        ranked, _ = rank_for_n(history, score_candidates(history, adjacency), 5, [], k=2)
        self.assertEqual([row[0] for row in ranked], ["hummus_recipe_2", "hummus_recipe_10"])


if __name__ == "__main__":
    unittest.main()
