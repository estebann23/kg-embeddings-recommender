"""Boundary checks for deterministic weighted path scoring."""

import math
import unittest

import numpy as np
from scipy import sparse

from src.recommenders.explicit_paths import rank, weighted_jaccard_profile


class ExplicitPathTests(unittest.TestCase):
    def test_idf_weighted_jaccard_and_positive_history_mean(self):
        binary = sparse.csr_matrix(np.array([[1, 1], [1, 0], [0, 1]], dtype=float))
        idf = np.log(4 / np.array([3, 3]))
        weighted = binary.multiply(idf).tocsr()
        mass = np.asarray(weighted.sum(axis=1)).ravel()
        scores = weighted_jaccard_profile((binary, weighted, mass, ["a", "b"]),
                                           [0, 1], np.array([2.0, 1.0]))
        self.assertTrue(math.isclose(scores[2], (2 * 0.5 + 0) / 3))
        self.assertTrue(math.isclose(scores[1], (2 * 0.5 + 1) / 3))

    def test_zero_evidence_uses_popularity_fallback_after_scored(self):
        ranked, scorable = rank(np.array([0.0, 0.2, 0.0, 0.0]), {0}, [2, 3, 1, 0], k=3)
        self.assertEqual(scorable, 1)
        self.assertEqual([entry[0] for entry in ranked], [1, 2, 3])
        self.assertEqual([entry[2] for entry in ranked], ["score", "fallback", "fallback"])

    def test_score_ties_use_numeric_recipe_order(self):
        ranked, _ = rank(np.array([0.0, 0.5, 0.5, 0.2]), {0}, [], k=2)
        self.assertEqual([entry[0] for entry in ranked], [1, 2])


if __name__ == "__main__":
    unittest.main()
