"""Metric boundary cases for the frozen RQ1 accuracy-only test pass."""

import math
import unittest

from src.recommenders.rq1_accuracy_evaluate import metric_values


class AccuracyMetricTests(unittest.TestCase):
    def test_precision_denominator_and_ndcg_ideal(self):
        ranking = [f"recipe_{index}" for index in range(50)]
        targets = {"recipe_0": 5, "recipe_12": 4, "recipe_1": 1,
                   "missing_positive": 5}
        result = metric_values(ranking, targets)
        self.assertEqual(result["positive_test_targets"], 3)
        self.assertEqual(result["hits_at_10"], 1)
        self.assertEqual(result["hits_at_50"], 2)
        self.assertEqual(result["precision_at_10"], 0.1)
        self.assertEqual(result["precision_at_50"], 0.04)
        ideal = 2 + 2 / math.log2(3) + 1 / math.log2(4)
        self.assertTrue(math.isclose(result["ndcg_at_10"], 2 / ideal))
        self.assertTrue(math.isclose(result["ndcg_at_50"],
                                     (2 + 1 / math.log2(14)) / ideal))

    def test_no_positive_target_has_zero_precision_and_undefined_ndcg(self):
        result = metric_values([f"recipe_{index}" for index in range(50)],
                               {"recipe_0": 1, "recipe_2": 3})
        self.assertEqual(result["precision_at_10"], 0)
        self.assertEqual(result["precision_at_50"], 0)
        self.assertIsNone(result["ndcg_at_10"])
        self.assertIsNone(result["ndcg_at_50"])


if __name__ == "__main__":
    unittest.main()
