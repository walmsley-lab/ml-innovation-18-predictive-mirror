"""Causality, determinism, and score boundaries for the frozen benchmark."""
import unittest
import numpy as np
from src import rgb_benchmark as b


class RGBBenchmarkTests(unittest.TestCase):
    def test_deterministic_world(self):
        first = b.frames(b.scenes(71000, 2, context=4, horizon=3, size=20))
        second = b.frames(b.scenes(71000, 2, context=4, horizon=3, size=20))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 7, 20, 20, 3))

    def test_no_future_leakage(self):
        x = b.frames(b.scenes(71000, 2, context=4, horizon=3, size=20))
        altered = x.copy()
        altered[:, 4:] = 1 - altered[:, 4:]
        a, c = b.controls(x[:, :4], 3), b.controls(altered[:, :4], 3)
        for key in a:
            np.testing.assert_array_equal(a[key], c[key])
        self.assertNotEqual(b.score(a["persistence"], x, 4, (3,))["3"]["mse"],
                            b.score(a["persistence"], altered, 4, (3,))["3"]["mse"])

    def test_exact_truth_and_invalid_horizon(self):
        x = b.frames(b.scenes(71100, 1, context=4, horizon=3, size=20))
        self.assertEqual(b.score(x[:, 4:], x, 4, (3,))["3"]["mse"], 0.)
        with self.assertRaises(ValueError):
            b.score(x[:, 4:], x, 4, (10,))
