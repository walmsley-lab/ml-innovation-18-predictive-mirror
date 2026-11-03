"""Independent hardware-constrained circuit-cell benchmark checks."""
import unittest

import numpy as np
from experiments.circuit_rgb_comparison import experiment
from src import rgb_circuit_cell as cell


class CircuitComparisonTests(unittest.TestCase):
    def test_circuit_locality_and_modes(self):
        state = np.zeros((1, 16, 16, 3, 2), np.float32)
        state[0, 8, 8, 0, 0] = 0.8
        drive = np.zeros((1, 16, 16, 3), np.float32)
        coupled = cell.update(state, drive)
        isolated = cell.update(state, drive, "no_coupling")
        self.assertGreater(float(coupled[0, 8, 9, 0, 0]), 0)
        self.assertEqual(float(isolated[0, 8, 9, 0, 0]), 0)
        self.assertEqual(float(coupled[0, 0, 0, 0, 0]), 0)
        quantized = cell.update(state, drive, "quantized_4bit")
        np.testing.assert_allclose(quantized * 7, np.round(quantized * 7), atol=1e-6)

    def test_portable_formula(self):
        state = np.zeros((1, 8, 8, 3, 2), np.float32)
        state[0, 4, 4, 0] = (0.6, 0.2)
        d = np.zeros((1, 8, 8, 3), np.float32)
        observed = cell.update(state, d)
        expected_v = (1 - .18) * .6 + .12 * (0 - .6) + .30 * .6 * .2
        expected_m = .2 + .04 * (np.tanh(2 * expected_v) - .2)
        self.assertAlmostEqual(float(observed[0, 4, 4, 0, 0]), expected_v, places=6)
        self.assertAlmostEqual(float(observed[0, 4, 4, 0, 1]), expected_m, places=6)

    def test_future_frames_never_enter_forecast(self):
        rng = np.random.default_rng(5)
        context = rng.random((2, 4, 16, 16, 3), dtype=np.float32)
        training = rng.random((2, 5, 16, 16, 3), dtype=np.float32)
        head = cell.fit_head(training, context=4, ticks=1)
        forecast = cell.rollout(context, head, 3, ticks=1)
        self.assertEqual(forecast.shape, (2, 3, 16, 16, 3))
        np.testing.assert_array_equal(forecast, cell.rollout(context.copy(), head, 3, ticks=1))
        np.testing.assert_array_equal(cell.rollout(context, np.zeros(4, np.float32), 3, ticks=1),
                                      np.repeat(context[:, -1:], 3, axis=1))

    def test_end_to_end_independent_splits_and_controls(self):
        r = experiment(seed=3, train_episodes=2, validation_episodes=1,
                       test_episodes=2, context=3, horizon=3, size=16, ticks=1)
        self.assertEqual(r["mode_metrics"]["full"]["3"]["episodes"], 2)
        self.assertIn("constant_velocity", r["mode_metrics"])
        self.assertNotEqual(r["train_seed_start"], r["test_seed_start"])
        self.assertTrue(np.isfinite(r["mode_metrics"]["full"]["1"]["mse"]))


if __name__ == "__main__":
    unittest.main()
