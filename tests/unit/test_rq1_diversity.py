"""Boundary checks for ingredient ILD and catalog exposure concentration."""

import math
import unittest

from src.recommenders.rq1_diversity import gini, list_ild


class DiversityMetricTests(unittest.TestCase):
    def test_pairwise_jaccard_excludes_unmapped_recipe(self):
        ingredients = {"a": frozenset({"salt", "tomato"}),
                       "b": frozenset({"tomato", "onion"})}
        value, possible, measurable = list_ild(["a", "b", "unmapped"], ingredients)
        self.assertEqual(possible, 3)
        self.assertEqual(measurable, 1)
        self.assertTrue(math.isclose(value, 2 / 3))

    def test_no_measurable_pair_is_undefined(self):
        value, possible, measurable = list_ild(["a", "unmapped"],
                                                {"a": frozenset({"salt"})})
        self.assertIsNone(value)
        self.assertEqual((possible, measurable), (1, 0))

    def test_gini_includes_zero_exposure_items(self):
        self.assertEqual(gini([1, 1, 1, 1]), 0)
        self.assertTrue(math.isclose(gini([0, 0, 1, 1]), 0.5))
        self.assertTrue(math.isclose(gini([0, 0, 0, 4]), 0.75))


if __name__ == "__main__":
    unittest.main()
