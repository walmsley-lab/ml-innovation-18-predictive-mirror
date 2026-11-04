"""Anti-collapse NCA: control parity, non-saturating head, foreground loss."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca
from src import rgb_grounded_nca as grounded


class GroundedNCATests(unittest.TestCase):
    def setUp(self):
        self.x = jnp.asarray(np.random.default_rng(0).random(
            (2, 7, 8, 8, 3), dtype=np.float32))

    def test_control_matches_baseline_exactly(self):
        params = nca.init(jax.random.PRNGKey(1), 4)
        base = np.asarray(nca.rollout(params, self.x[:, :4], 3, 2))
        same = np.asarray(grounded.rollout(params, self.x[:, :4], 3, 2, "residual"))
        np.testing.assert_allclose(same, base, atol=1e-6)

    def test_blend_head_stays_in_range_and_can_leave_persistence(self):
        params = grounded.init(jax.random.PRNGKey(2), 4, output="blend")
        self.assertIn("gate_readout", params)
        out = np.asarray(grounded.rollout(params, self.x[:, :4], 5, 2, "blend"))
        self.assertTrue(np.isfinite(out).all())
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)
        # With a strongly positive gate the output follows the colour head, i.e.
        # it is NOT pinned to the fed-back image: the blend can move off-rail.
        loud = dict(params, gate_bias=jnp.full((1,), 6.0, jnp.float32),
                    rgb_bias=jnp.full((3,), 3.0, jnp.float32))
        bright = np.asarray(grounded.rollout(loud, self.x[:, :4], 3, 2, "blend"))
        self.assertGreater(bright.mean(), 0.3)  # not collapsed to black

    def test_foreground_weight_penalizes_background_prediction(self):
        # Black context, bright foreground only in the FUTURE, so a near-black
        # persistence prediction is wrong exactly on the foreground.
        target = np.zeros((1, 6, 8, 8, 3), np.float32)
        target[:, 3:, 2:5, 2:5] = 0.9  # foreground appears in the future frames
        params = nca.init(jax.random.PRNGKey(3), 4)
        # Freeze the model to a near-black predictor by zeroing the readout, so
        # the loss difference is driven only by the foreground weighting term.
        params = dict(params, rgb_readout=jnp.zeros((4, 3)), rgb_bias=jnp.zeros((3,)))
        plain = float(grounded.loss(params, jnp.asarray(target), 3, 3, 2,
                                    "residual", foreground_weight=0.0))
        weighted = float(grounded.loss(params, jnp.asarray(target), 3, 3, 2,
                                       "residual", foreground_weight=6.0))
        self.assertGreater(weighted, plain)

    def test_rejects_bad_output_mode(self):
        with self.assertRaises(ValueError):
            grounded.init(jax.random.PRNGKey(4), 4, output="nope")
        with self.assertRaises(ValueError):
            grounded.rollout(nca.init(jax.random.PRNGKey(4), 4),
                             self.x[:, :4], 3, 2, "nope")


if __name__ == "__main__":
    unittest.main()
