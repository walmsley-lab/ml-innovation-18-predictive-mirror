"""Multi-horizon skeleton graph NCA: causal dream, controls and CPU smoke."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.graph_skeleton_nca import examples
from experiments.graph_skeleton_horizons import last_seen, loss, rollout, run, score
from src import graph_skeleton_nca as nca


class GraphSkeletonHorizonTests(unittest.TestCase):
    def test_untrained_dream_is_last_seen(self):
        observed, visibility, _ = examples((8, 9), steps=15)
        weights = nca.init(jax.random.PRNGKey(5), channels=4)
        prediction = np.asarray(rollout(weights, jnp.asarray(observed[:, :8]),
                                       jnp.asarray(visibility[:, :8]),
                                       8, nca.adjacency()))
        baseline = last_seen(observed[:, :8], visibility[:, :8], 8)
        np.testing.assert_allclose(prediction, baseline, atol=1e-6)

    def test_context_only_and_hidden_labels_not_used(self):
        observed, mask, truth = examples((8,), steps=15)
        weights = nca.init(jax.random.PRNGKey(1), channels=4)
        graph = nca.adjacency()
        forecast = np.asarray(rollout(weights, jnp.asarray(observed[:, :8]),
                                      jnp.asarray(mask[:, :8]), 8, graph))
        changed = observed.copy()
        changed[:, 8:] = .8
        np.testing.assert_array_equal(forecast, np.asarray(
            rollout(weights, jnp.asarray(changed[:, :8]),
                    jnp.asarray(mask[:, :8]), 8, graph)))
        clean = float(loss(weights, jnp.asarray(observed), jnp.asarray(mask),
                           8, 8, graph, True))
        changed = observed.copy()
        changed[mask[..., 0] == 0] = 1.
        altered = float(loss(weights, jnp.asarray(changed), jnp.asarray(mask),
                             8, 8, graph, True))
        self.assertAlmostEqual(clean, altered, places=6)

    def test_reappearance_and_three_controls(self):
        observed, mask, truth = examples((20, 21), steps=17,
                                         occlusion_frames=3)
        baseline = last_seen(observed[:, :8], mask[:, :8], 10)
        velocity = last_seen(observed[:, :8], mask[:, :8], 10, True)
        self.assertEqual(baseline.shape, (2, 10, len(nca.JOINTS), 2))
        self.assertTrue(np.isfinite(velocity).all())
        result = score(baseline, truth[:, 8:18], mask[:, 8:18], mask[:, 7],
                       (1, 3, 5, 10))
        self.assertIn("reappearance_events", result["5"])

    def test_one_step_training_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(folder, seeds=(3,), steps=1, batch_size=1, heldout=2,
                         context=8, horizon=10, channels=4)
            self.assertEqual(set(result["runs"][0]["metrics"]),
                             {"graph", "no_messages", "last_seen", "constant_velocity"})
            self.assertTrue((Path(folder)/"seed-3-graph.npz").exists())
            self.assertTrue((Path(folder)/"metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
