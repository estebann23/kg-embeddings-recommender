"""Negative samples preserve graph types and never relabel known ratings."""

import unittest

try:
    import torch
    from src.recommenders.rotate_sampling import TypedTailNegativeSampler
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyKEEN extra is not installed")
class SamplingProtocolTests(unittest.TestCase):
    def test_rating_samples_exclude_all_observed_star_relations(self):
        # Entity IDs: 0 user; 1,2,3 recipes; 4 ingredient.
        known = torch.tensor([[0, 0, 1], [0, 1, 2], [1, 2, 4]])
        sampler = TypedTailNegativeSampler(
            mapped_triples=known,
            relation_tail_ids={0: [1, 2, 3], 1: [1, 2, 3], 2: [4, 5]},
            rating_relation_ids=[0, 1], num_negs_per_pos=3,
        )
        negatives, mask = sampler.sample(known)
        self.assertIsNone(mask)
        self.assertEqual(negatives.shape, (3, 3, 3))
        self.assertEqual(negatives[0, :, 2].tolist(), [3, 3, 3])
        self.assertEqual(negatives[1, :, 2].tolist(), [3, 3, 3])
        self.assertEqual(negatives[2, :, 2].tolist(), [5, 5, 5])
        self.assertTrue(torch.equal(negatives[:, :, :2], known[:, None, :2].expand(-1, 3, -1)))

    def test_no_admissible_negative_fails(self):
        known = torch.tensor([[0, 0, 1], [0, 1, 2]])
        sampler = TypedTailNegativeSampler(
            mapped_triples=known, relation_tail_ids={0: [1, 2], 1: [1, 2]},
            rating_relation_ids=[0, 1],
        )
        with self.assertRaises(ValueError):
            sampler.sample(known)


if __name__ == "__main__":
    unittest.main()
