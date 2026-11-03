"""Toy local recovery must be bounded, causal, and independently trainable."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca
from src import rgb_homeostatic_nca as recovery
from experiments.rgb_homeostatic_recovery import run


class HomeostaticRecoveryTests(unittest.TestCase):
    def test_plain_and_zero_strength_match_existing_rollout(self):
        x = jnp.asarray(np.random.default_rng(4).random(
            (1, 5, 8, 8, 3), dtype=np.float32))
        params = nca.init(jax.random.PRNGKey(1), 4)
        params = dict(params, rgb_readout=jnp.ones((4, 3)) * .04)
        baseline = np.asarray(nca.rollout(params, x, 3, 1))
        for mode in recovery.MODES:
            strength = 0 if mode != "plain" else .4
            got = np.asarray(recovery.rollout(
                params, x, 3, 1, mode, strength))
            np.testing.assert_allclose(got, baseline, atol=1e-6)

    def test_bounded_regulation_and_no_future_peeking(self):
        x = np.random.default_rng(6).random((1, 7, 8, 8, 3),
                                           dtype=np.float32)
        params = nca.init(jax.random.PRNGKey(2), 4)
        params = dict(params, rgb_readout=jnp.ones((4, 3)) * .04)
        prefix = jnp.asarray(x[:, :4])
        for mode in recovery.MODES:
            forecast, diag = recovery.rollout_and_diagnostics(
                params, prefix, 3, 1, mode, .3, .7)
            self.assertTrue(np.isfinite(np.asarray(forecast)).all())
            self.assertGreaterEqual(float(diag["max_recovery"]), 0)
            self.assertLessEqual(float(diag["max_recovery"]), 1 + 1e-6)
            altered = x.copy()
            altered[:, 4:] = 0
            np.testing.assert_array_equal(
                np.asarray(forecast),
                np.asarray(recovery.rollout(
                    params, jnp.asarray(altered[:, :4]), 3, 1,
                    mode, .3, .7)))

    def test_recovery_changes_dynamics_without_new_parameters(self):
        params = nca.init(jax.random.PRNGKey(3), 4)
        hidden = jnp.ones((1, 8, 8, 4))
        image = jnp.zeros((1, 8, 8, 3))
        surprise = jnp.zeros_like(image)
        r = jnp.zeros_like(hidden)
        normal, _ = recovery.cell_tick(
            params, hidden, r, image, surprise, "plain", .7, .5)
        regulated, next_r = recovery.cell_tick(
            params, hidden, r, image, surprise, "local_recovery", .7, .5)
        self.assertGreater(float(jnp.abs(normal - regulated).mean()), 0)
        self.assertGreater(float(next_r.max()), 0)
        self.assertLessEqual(float(next_r.max()), 1)

    def test_spatial_roughness_is_nontrivial(self):
        x = np.linspace(-1, 1, 8, dtype=np.float32)
        smooth = jnp.asarray(np.broadcast_to(x[None, None, :, None],
                                              (1, 8, 8, 1)))
        cells = np.indices((8, 8)).sum(axis=0) % 2
        checker = jnp.asarray(cells[None, :, :, None], jnp.float32)
        self.assertGreater(float(recovery.spatial_roughness(checker)),
                           float(recovery.spatial_roughness(smooth)))
        self.assertEqual(float(recovery.spatial_roughness(
            jnp.ones_like(checker))), 0.0)

    def test_minimal_train_and_evaluate_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run(directory, seeds=(3,), steps=1, context=4,
                         horizon=2, size=16, ticks=1, channels=4,
                         batch_size=1)
            self.assertEqual(set(report["results"][0]["arms"]),
                             set(recovery.MODES))
            for arm in report["results"][0]["arms"].values():
                self.assertIn("mean_spatial_roughness", arm["activity"])
                self.assertLessEqual(arm["validation_mse"],
                                     arm["step_zero_validation_mse"])
            self.assertTrue((Path(directory) / "metrics.json").exists())
            self.assertTrue((Path(directory) / "seed-3-local_recovery.npz").exists())


if __name__ == "__main__":
    unittest.main()
