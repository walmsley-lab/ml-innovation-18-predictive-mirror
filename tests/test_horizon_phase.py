import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.horizon_phase import ARMS, horizon_loss, rollout_one, run
from src import video_residual


class HorizonPhaseTests(unittest.TestCase):
    def test_initial_forecast_is_persistence_for_every_arm(self):
        frames = np.random.default_rng(12).random((8, 12, 12, 1), dtype=np.float32)
        params = video_residual.init(jax.random.PRNGKey(5))
        for arm in ARMS:
            predicted = np.asarray(rollout_one(
                params, jnp.asarray(frames), 5, arm, 0.25))
            self.assertEqual(predicted.shape, (5, 12, 12, 1))
            np.testing.assert_allclose(
                predicted, np.broadcast_to(frames[-1], predicted.shape), atol=1e-7)

    def test_horizon_training_uses_future_as_labels_not_rollout_inputs(self):
        params = video_residual.init(jax.random.PRNGKey(9))
        rng = np.random.default_rng(3)
        history = rng.random((8, 12, 12, 1), dtype=np.float32)
        future = rng.random((10, 12, 12, 1), dtype=np.float32)
        pred = np.asarray(rollout_one(params, jnp.asarray(history), 10,
                                      "phase_coupled", 0.25))
        tampered = future.copy()
        tampered[-1] = 0
        np.testing.assert_array_equal(pred, np.asarray(rollout_one(
            params, jnp.asarray(history), 10, "phase_coupled", 0.25)))
        examples = jnp.asarray(np.concatenate((history, future))[None])
        first = float(horizon_loss(params, examples, 8, 10, "plain", 0.25))
        examples = jnp.asarray(np.concatenate((history, tampered))[None])
        second = float(horizon_loss(params, examples, 8, 10, "plain", 0.25))
        self.assertNotEqual(first, second)

    def test_training_smoke_three_independent_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            recording = Path(folder) / "recording.npz"
            np.savez_compressed(recording,
                                frames=np.random.default_rng(22).random(
                                    (200, 12, 12, 1), dtype=np.float32))
            result = run(recording, Path(folder) / "runs", seed=5,
                         steps=2, batch_size=2, context=4, horizon=5)
            self.assertEqual(set(result["arms"]), set(ARMS))
            self.assertEqual(result["forecast_frames"], 5)
            for arm in ARMS:
                self.assertTrue((Path(folder) / "runs" / f"{arm}.npz").exists())
                self.assertIn("5", result["arms"][arm])


if __name__ == "__main__":
    unittest.main()
