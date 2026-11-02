"""Gallery determinism, causality and rendering tests."""
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np

from experiments.rgb_prediction_gallery import predict_from_context, render, run
from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark


class RGBGalleryTests(unittest.TestCase):
    def test_no_future_frames_enter_predictor(self):
        frames = np.zeros((2, 7, 16, 16, 3), np.float32)
        seen = []
        def predictor(prefix):
            seen.append(prefix.copy())
            return prefix[:, -1]
        predict_from_context(predictor, frames, 4)
        frames[:, 4:] = 1
        predict_from_context(predictor, frames, 4)
        self.assertEqual(seen[0].shape, (2, 4, 16, 16, 3))
        np.testing.assert_array_equal(seen[0], seen[1])

    def test_frozen_untrained_checkpoint_matches_persistence(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow is optional")
        with tempfile.TemporaryDirectory() as folder:
            params = nca.init(jax.random.PRNGKey(3), channels=4)
            checkpoint = Path(folder)/"checkpoint.npz"
            np.savez_compressed(checkpoint, **jax.tree.map(np.asarray, params))
            report = run(checkpoint, Path(folder)/"out", seed=71500,
                         episodes=1, context=4, horizon=3, size=16, ticks=1,
                         scale=2)
            self.assertTrue(Path(report["images"][0]).exists())
            self.assertEqual(report["leads"], [1, 3])
            for lead in ("1", "3"):
                self.assertAlmostEqual(report["metrics"]["nca"][lead]["mse"],
                                       report["metrics"]["persistence"][lead]["mse"],
                                       places=6)

    def test_render_tiles_include_future_truth_separately(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional")
        frames = benchmark.frames(benchmark.scenes(
            71200, 1, context=4, horizon=3, size=16))[0]
        predictions = np.repeat(frames[3:4], 3, axis=0)
        controls = {"persistence": predictions, "constant_velocity": predictions}
        activity = np.zeros((3, 16, 16, 1), np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"gallery.png"
            render(frames, predictions, activity, controls, 4, (1, 3), path, 2)
            with Image.open(path) as image:
                self.assertEqual(image.size, (56+7*32, 38+2*(32+22)))


if __name__ == "__main__":
    unittest.main()
