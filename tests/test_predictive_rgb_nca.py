"""Focused RGB NCA tests; OpenCV is optional for core CPU CI."""
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None
import jax
import jax.numpy as jnp
import numpy as np

from experiments.rgb_nca import fit, preprocess_color, read_recording
from experiments.rgb_nca_viewer import render_grid, score
from src import predictive_rgb_nca as nca


class RGBPredictiveNCATests(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "OpenCV optional")
    def test_three_channel_color_is_preserved(self):
        bgr = np.zeros((20, 20, 3), np.uint8)
        bgr[:] = (0, 0, 255)
        rgb = preprocess_color(bgr, 16, cv2)
        np.testing.assert_allclose(rgb[0, 0], [1., 0., 0.])
        self.assertEqual(rgb.shape, (16, 16, 3))

    def test_initialized_dreams_are_persistence_and_causal(self):
        params = nca.init(jax.random.PRNGKey(3), channels=8)
        observed = np.random.default_rng(8).random((1, 5, 16, 16, 3), dtype=np.float32)
        forecasts, activity = nca.rollout_and_activity(
            params, jnp.asarray(observed), 5, ticks=2)
        self.assertEqual(forecasts.shape, (1, 5, 16, 16, 3))
        self.assertEqual(activity.shape, (1, 5, 16, 16, 1))
        np.testing.assert_allclose(
            np.asarray(forecasts),
            np.broadcast_to(observed[:, -1:], forecasts.shape),
            atol=1e-7)
        self.assertTrue(np.isfinite(np.asarray(activity)).all())

    def test_observe_uses_prediction_error(self):
        params = nca.init(jax.random.PRNGKey(2), channels=8)
        original = nca.initial_state(params, 1, 16, 16)
        red = jnp.broadcast_to(jnp.array([1., 0., 0.]), (1, 16, 16, 3))
        blue = jnp.broadcast_to(jnp.array([0., 0., 1.]), (1, 16, 16, 3))
        red_state = nca.observe(params, original, red, ticks=2)
        blue_state = nca.observe(params, original, blue, ticks=2)
        self.assertGreater(float(jnp.max(jnp.abs(red_state[0] - blue_state[0]))), 0)
        predicted_state, prediction = nca.dream(params, red_state, ticks=2)
        np.testing.assert_allclose(np.asarray(prediction), np.asarray(red), atol=1e-7)
        # Same new observation but different prior prediction -> different correction.
        corrected = nca.observe(params, predicted_state, blue, ticks=2)
        unpredicted = nca.observe(params, (predicted_state[0], blue), blue, ticks=2)
        self.assertGreater(float(jnp.max(jnp.abs(corrected[0] - unpredicted[0]))), 0)

    def test_model_has_nonzero_readout_training_gradient(self):
        params = nca.init(jax.random.PRNGKey(6), channels=8)
        samples = jnp.asarray(np.random.default_rng(10).random(
            (1, 9, 16, 16, 3), dtype=np.float32))
        gradient = jax.grad(nca.loss)(params, samples, 5, 4, 2)
        self.assertGreater(float(jnp.abs(gradient["rgb_readout"]).sum()), 0.)

    def test_rejects_grayscale_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            np.savez_compressed(path, frames=np.zeros((120, 16, 16, 1), np.float32))
            with self.assertRaises(ValueError):
                read_recording(path)

    def test_short_training_and_holdout_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.npz"
            frames = np.random.default_rng(10).random(
                (140, 16, 16, 3), dtype=np.float32)
            np.savez_compressed(path, frames=frames)
            report = fit(path, Path(directory) / "out", seed=3, steps=2,
                         batch_size=1, context=4, horizon=3, ticks=1, channels=6)
            self.assertEqual(report["architecture"], "predictive_rgb_nca")
            self.assertEqual(report["forecast_horizon_frames"], 3)
            self.assertTrue((Path(directory) / "out" / "parameters.npz").exists())
            self.assertIn("3", report["horizons"])

    @unittest.skipIf(cv2 is None, "OpenCV optional")
    def test_rgb_render_and_metric(self):
        red = np.zeros((16, 16, 3), np.float32)
        red[..., 0] = 1
        forecast = np.stack([red, red, red])
        activity = np.zeros((3, 16, 16, 1), np.float32)
        canvas = render_grid(red, forecast, activity, (1, 3), cv2,
                             actuals={1: red, 3: red}, panel_size=80)
        self.assertEqual(canvas.shape, (45+160, 75+400, 3))
        self.assertEqual(score(red, red, red)["model_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
