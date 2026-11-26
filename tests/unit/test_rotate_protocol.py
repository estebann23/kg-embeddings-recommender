"""Protocol checks that do not train or initialize a RotatE model."""

import unittest

from src.recommenders.rotate_ranking import rank_candidates, select_weight


class RankingProtocolTests(unittest.TestCase):
    def test_unscorable_fallback_and_training_exclusion(self):
        scores = {"RATED_4": {"a": 3.0, "b": 2.0},
                  "RATED_5": {"a": 1.0, "b": 4.0}}
        self.assertEqual(rank_candidates(["missing", "b", "a", "trained"], {"trained"}, scores, 1),
                         ["b", "a", "missing"])

    def test_validation_selects_five_star_weight(self):
        users = [{"training_recipes": set(),
                  "scores": {"RATED_4": {"a": 3.0, "b": 2.0},
                             "RATED_5": {"a": 1.0, "b": 4.0}},
                  "validation_ratings": {"b": 5}}]
        result = select_weight(users, ["a", "b", "missing"])
        self.assertEqual(result["weight_five"], 0.75)
        self.assertEqual(result["users_with_positive_validation"], 1)

    def test_inconsistent_relation_coverage_fails(self):
        with self.assertRaises(ValueError):
            rank_candidates(["a", "b"], set(),
                            {"RATED_4": {"a": 1}, "RATED_5": {"b": 1}}, 0.5)


if __name__ == "__main__":
    unittest.main()
