"""Frozen RGB NCA trace adaptation: causality, controls, and smoke."""
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as base, rgb_error_trace as trace
from experiments.rgb_error_trace import run


class RGBErrorTraceTests(unittest.TestCase):
    def setUp(self):
        self.images = np.random.default_rng(12).random(
            (1, 7, 16, 16, 3), dtype=np.float32)
        self.params = base.init(jax.random.PRNGKey(11), 4)

    def test_frozen_is_original_and_causal(self):
        x = jnp.asarray(self.images[:, :4])
        reference = np.asarray(base.rollout(self.params, x, 3, 1))
        unchanged = np.asarray(trace.rollout(self.params, x, 3, 1, "frozen"))
        np.testing.assert_allclose(reference, unchanged, atol=1e-6)
        altered = self.images.copy()
        altered[:, 4:] = 0.
        np.testing.assert_array_equal(unchanged, np.asarray(trace.rollout(
            self.params, jnp.asarray(altered[:, :4]), 3, 1, "frozen")))

    def test_local_trace_uses_observed_surprise(self):
        x = jnp.asarray(self.images[:, :4])
        plain = np.asarray(trace.rollout(self.params, x, 3, 1, "frozen"))
        adapted = np.asarray(trace.rollout(
            self.params, x, 3, 1, "local_trace", strength=.25))
        shuffled = np.asarray(trace.rollout(
            self.params, x, 3, 1, "shuffled_trace", strength=.25))
        self.assertGreater(float(np.abs(adapted-plain).mean()), 0.)
        self.assertGreater(float(np.abs(adapted-shuffled).mean()), 0.)
        np.testing.assert_allclose(np.asarray(trace.rollout(
            self.params, x, 3, 1, "local_trace", strength=0.)), plain)

    def test_first_frame_is_not_prediction_error(self):
        # Untrained model on a static scene predicts perfectly after the
        # first frame, so no prediction error exists for the trace to store.
        still = jnp.asarray(np.repeat(self.images[:, :1], 5, axis=1))
        frozen = np.asarray(trace.rollout(self.params, still, 3, 1, "frozen"))
        local = np.asarray(trace.rollout(self.params, still, 3, 1, "local_trace"))
        np.testing.assert_allclose(local, frozen, atol=1e-7)

    def test_frozen_checkpoint_report(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "parameters.npz"
            np.savez_compressed(checkpoint, **jax.tree.map(np.asarray, self.params))
            result = run(checkpoint, Path(folder)/"result", episodes=1,
                         context=4, horizon=3, size=16, ticks=1)
            self.assertEqual(set(result["arms"]), set(trace.MODES))
            self.assertTrue((Path(folder)/"result"/"example.npz").exists())
            self.assertEqual(result["arms"]["frozen"]
                             ["mean_absolute_difference_from_frozen"], 0.)


if __name__ == "__main__":
    unittest.main()
