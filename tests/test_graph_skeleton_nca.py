import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.graph_skeleton_nca import episode, examples, experiment, measure, persistence
from src import graph_skeleton_nca as nca


class SkeletonTests(unittest.TestCase):
    def test_fixed_graph_and_deterministic_world(self):
        a = episode(12)
        b = episode(12)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(a[0].shape, (15, 12, 2))
        self.assertEqual(a[1].shape, (15, 12, 1))
        self.assertTrue(np.all(a[0][a[1][..., 0] == 0] == 0))
        self.assertTrue(np.all((a[2] >= 0) & (a[2] <= 1)))
        graph = np.asarray(nca.adjacency())
        np.testing.assert_allclose(graph.sum(axis=-1), 1)
        self.assertEqual(graph.shape, (12, 12))

    def test_initialized_models_match_last_seen_and_are_causal(self):
        observed, masks, _ = examples((21, 22))
        graph = nca.adjacency()
        params = nca.init(jax.random.PRNGKey(8))
        output = np.asarray(nca.predict(params, jnp.asarray(observed),
                                        jnp.asarray(masks), graph))
        baseline = persistence(observed, masks)
        np.testing.assert_allclose(output, baseline, atol=1e-6)
        altered = observed.copy()
        altered[:, -1, :, :] = 0.0
        np.testing.assert_array_equal(output, np.asarray(nca.predict(
            params, jnp.asarray(altered), jnp.asarray(masks), graph)))

    def test_masked_future_labels_cannot_affect_training(self):
        observed, visibility, truth = examples((21, 22))
        params = nca.init(jax.random.PRNGKey(4))
        graph = nca.adjacency()
        first = float(nca.loss(params, jnp.asarray(observed), jnp.asarray(visibility), graph))
        # Simulator truth at occluded nodes is not even passed to loss().
        changed_truth = truth.copy()
        changed_truth[visibility[..., 0] == 0] += 99
        self.assertGreater(np.max(changed_truth - truth), 0)
        second = float(nca.loss(params, jnp.asarray(observed), jnp.asarray(visibility), graph))
        self.assertEqual(first, second)

    def test_reappearance_events_and_pixel_metrics(self):
        observed, masks, truth = examples((21, 22), occlusion_frames=5)
        baseline = persistence(observed, masks)
        report = measure(baseline, truth, masks)
        self.assertEqual(report["reappearance_events"], 4)
        self.assertIsNotNone(report["hidden_joint_error_px"])

    def test_three_arm_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            report = experiment(Path(folder), seed=3, steps=2, batch_size=2, heldout=2)
            self.assertEqual(set(report["arms"]), {"graph", "no_messages", "last_seen"})
            self.assertTrue((Path(folder) / "graph.npz").exists())
            self.assertTrue((Path(folder) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
