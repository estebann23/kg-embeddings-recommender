"""Boundary checks for condition-2 flat-attribute item CF."""

import math
import unittest

import numpy as np
from scipy import sparse

from src.recommenders.item_cf_attributes import jaccard_rows, rank_scores


class AttributeItemCFTests(unittest.TestCase):
    def test_jaccard_zero_for_missing_and_direct_overlap(self):
        matrix = sparse.csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.int32))
        sizes = np.asarray(matrix.getnnz(axis=1), dtype=np.float64)
        actual = jaccard_rows(matrix, [0], sizes)
        self.assertTrue(math.isclose(actual[0, 1], 1 / 3))
        self.assertEqual(actual[0, 2], 0)

    def test_negative_rating_and_fallback_are_preserved(self):
        # Rows are history recipes in numeric-ID order; columns are candidate recipes.
        similarities = np.array([[0, 0, 0.8, 0], [0, 0, 0.2, 0]], dtype=np.float64)
        ranked, scorable = rank_scores(similarities, np.array([-1.0, 1.0]),
                                       np.array([0, 1]), [3, 2, 0], 5, k=2)
        self.assertEqual(scorable, 1)
        self.assertEqual([row[0] for row in ranked], [2, 3])
        self.assertEqual([row[2] for row in ranked], ["score", "fallback"])
        self.assertTrue(math.isclose(ranked[0][1], -0.6))

    def test_top_neighbor_tie_uses_earlier_history_recipe(self):
        similarities = np.array([[0, 0, 0.5], [0, 0, 0.5]], dtype=np.float64)
        ranked, _ = rank_scores(similarities, np.array([-1.0, 1.0]),
                                    np.array([0, 1]), [2], 1, k=1)
        self.assertEqual(ranked[0][0], 2)
        self.assertEqual(ranked[0][1], -1.0)


if __name__ == "__main__":
    unittest.main()
