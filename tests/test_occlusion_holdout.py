import unittest
import jax
import numpy as np
from src import model
from src.world import sequence
from experiments.occlusion_holdout import benchmark


class ExtendedOcclusionTests(unittest.TestCase):
    def test_default_width_matches_explicit_unit_wall(self):
        for seed in (0, 13):
            a = sequence(seed, steps=10)
            b = sequence(seed, steps=10, wall_width=1)
            np.testing.assert_array_equal(a[0], b[0])
            for k in a[2]:
                np.testing.assert_array_equal(a[2][k], b[2][k])

    def test_wide_wall_and_hidden_mask(self):
        frames, actions, hidden = sequence(8, steps=24, camera=False, wall_width=5)
        self.assertEqual(int((frames[0, 0, :, 0] == 0.35).sum()), 5)
        self.assertTrue(hidden["occluded"].any())
        self.assertFalse(np.any(hidden["occluded"] & hidden["visible_mask"].any(axis=(1, 2, 3))))
        with self.assertRaises(ValueError):
            sequence(8, wall_width=0)

    def test_width_specific_metrics_have_event_coverage(self):
        params = model.init(jax.random.PRNGKey(5))
        result = benchmark(params, "recurrent", [1, 3], [60_000_000], steps=20)
        self.assertEqual(set(result), {"1", "3"})
        for scores in result.values():
            self.assertEqual(set(scores), {"model", "persistence", "velocity"})
            self.assertIn("reappearance_count", scores["model"])


if __name__ == "__main__":
    unittest.main()
