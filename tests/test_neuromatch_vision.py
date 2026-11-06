import unittest
import jax
import numpy as np
from src import neuromatch_vision as m
from src import predictive_rgb_nca as nca


class NeuromatchVisionTests(unittest.TestCase):
    def test_sparse_budget_and_no_future(self):
        old = np.zeros((4, 4, 3))
        observed = old.copy()
        observed[0, 1] = 1
        observed[2, 2] = 0.5
        result, cost = m.sparse_observe(old, observed, max_fraction=1/16)
        self.assertEqual(cost["updated_pixels"], 1)
        np.testing.assert_array_equal(result[0, 1], np.ones(3))
        np.testing.assert_array_equal(result[2, 2], np.zeros(3))
        self.assertEqual(cost["measured_energy_joules"], None)

    def test_normalization_is_spatial_and_deterministic(self):
        frame = np.random.default_rng(9).uniform(0.1, 0.5, (6, 7, 3))
        np.testing.assert_array_equal(m.normalize_rgb(frame), m.normalize_rgb(frame.copy()))
        self.assertEqual(m.normalize_rgb(frame).shape, frame.shape)
        with self.assertRaises(ValueError):
            m.normalize_rgb(frame, radius=-1)

    def test_velocity_cka_rotations_and_time_shuffle(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(30, 4)).cumsum(axis=0)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        self.assertAlmostEqual(m.velocity_cka(a, a @ q), 1.0, places=6)
        self.assertLess(m.velocity_cka(a, a[::-1] @ q), 0.9)

    def test_targeted_lesion_shape_and_causal_selection(self):
        params = nca.init(jax.random.PRNGKey(2), channels=4)
        rng = np.random.default_rng(11)
        context = rng.uniform(0, 1, (1, 3, 8, 8, 3)).astype("float32")
        truth = np.zeros((1, 2, 8, 8, 3), dtype="float32")
        first = m.targeted_channel_lesions(params, context, truth, horizon=2)
        second = m.targeted_channel_lesions(params, context, np.ones_like(truth), horizon=2)
        self.assertEqual(first["targeted_channels"], second["targeted_channels"])
        self.assertFalse(first["selection_uses_future_truth"])
        self.assertTrue(np.isfinite(first["intact_mse"]))
        self.assertEqual(len(first["targeted_channels"]), 1)
        self.assertNotEqual(set(first["targeted_channels"]), set(first["random_channels"]))
        with self.assertRaises(ValueError):
            m.targeted_channel_lesions(params, context, truth, horizon=2, fraction=1.)
