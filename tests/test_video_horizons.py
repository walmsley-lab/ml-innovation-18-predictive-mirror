import unittest

import jax
import jax.numpy as jnp
import numpy as np

from experiments.video_horizons import compare, rollout
from src.video_residual import init


class VideoHorizonTests(unittest.TestCase):
    def test_rollout_persistence_at_initialization_and_causality(self):
        params = init(jax.random.PRNGKey(3))
        frames = np.random.default_rng(2).random((8, 12, 12, 1), dtype=np.float32)
        result = np.asarray(rollout(params, jnp.asarray(frames), 5))
        np.testing.assert_allclose(result, np.broadcast_to(frames[-1], result.shape), atol=1e-7)
        self.assertEqual(result.shape, (5, 12, 12, 1))

    def test_future_is_used_only_for_scoring(self):
        params = init(jax.random.PRNGKey(2))
        frames = np.random.default_rng(3).random((160, 12, 12, 1), dtype=np.float32)
        output = compare(params, frames, history=8, horizons=(1, 3))
        self.assertEqual(output["nonoverlapping_windows"], 2)
        for item in output["horizons"].values():
            self.assertAlmostEqual(item["model_mse"], item["persistence_mse"], places=7)

    def test_rejects_unscorable_horizon(self):
        params = init(jax.random.PRNGKey(2))
        with self.assertRaises(ValueError):
            compare(params, np.zeros((60, 12, 12, 1), np.float32), horizons=(20,))


if __name__ == "__main__":
    unittest.main()
