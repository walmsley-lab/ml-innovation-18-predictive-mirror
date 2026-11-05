"""Regression tests for candidate ranking and strict observation-only inputs."""
import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from experiments import contrastive_future_recognition as m


class ContrastiveFutureTests(unittest.TestCase):
    def test_all_candidates_same_start_and_recoverable_motion(self):
        history, choices, label = m.scene(11)
        self.assertEqual(history.shape, (8, 32, 32, 3))
        self.assertEqual(choices.shape, (4, 4, 32, 32, 3))
        self.assertTrue(0 <= label < 4)
        self.assertEqual(m.features(history).shape, (192,))
        np.testing.assert_allclose(choices[:, 0].sum(axis=(1, 2, 3)),
                                   choices[:, 0].sum(axis=(1, 2, 3))[0], rtol=.1)

    def test_future_does_not_enter_state_features(self):
        past, candidates, _ = m.scene(4)
        first = m.features(past)
        mutated = candidates.copy()
        mutated[:] = 0
        np.testing.assert_array_equal(first, m.features(past))
        self.assertFalse(np.array_equal(m.features(candidates), m.features(mutated)))

    def test_permutation_equivariance_and_gradients(self):
        past, options, _ = m.dataset(31000, 3)
        state, future = jnp.asarray(m.features(past)), jnp.asarray(m.features(options))
        init = m.initialize(jax.random.PRNGKey(1))
        a = m.tower_scores(init, state, future)
        order = np.array([2, 0, 3, 1])
        b = m.tower_scores(init, state, future[:, order])
        np.testing.assert_allclose(np.asarray(a)[:, order], np.asarray(b), atol=1e-5)
        _, grad = jax.value_and_grad(m.loss)(
            {k: init[k] for k in ("state_w", "state_b", "future_w", "future_b")},
            state, future, jnp.zeros(len(state), dtype=jnp.int32), "contrastive")
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in grad.values()))

    def test_smoke_train_and_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            report = m.run(root + "/metrics.json", seeds=(2,), steps=2,
                           train_count=6, test_count=4)
            self.assertEqual(report["data"]["test"], 4)
            self.assertIn("constant_velocity", report["controls"])
            self.assertIn("contrastive", report["runs"][0]["arms"])
            self.assertIn("supervised_mlp", report["runs"][0]["arms"])
            for entry in report["runs"][0]["arms"].values():
                self.assertTrue(0 <= entry["accuracy"] <= 1)


if __name__ == "__main__":
    unittest.main()
