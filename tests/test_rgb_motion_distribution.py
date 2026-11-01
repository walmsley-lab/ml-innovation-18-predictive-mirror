"""Distribution-training experiment: deterministic scenes, causal controls, tiny CPU smoke."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.rgb_motion_distribution import (
    SCENE_TYPES, baselines, center, constant_velocity, episode, measure,
    run, scenes, shift,
)


class RGBMotionDistributionTests(unittest.TestCase):
    def test_scenes_are_reproducible_and_have_distinct_families(self):
        a, kind = episode(31000, frames=14, size=24)
        b, again = episode(31000, frames=14, size=24)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(kind, again)
        self.assertIn(kind, SCENE_TYPES)
        frames, kinds = scenes(31000, 12, 14, 24)
        self.assertEqual(frames.shape, (12, 14, 24, 24, 3))
        self.assertEqual(len(kinds), 12)
        self.assertGreater(float(np.abs(frames[0]-frames[1]).sum()), .01)
        self.assertGreaterEqual(float(frames.min()), 0)
        self.assertLessEqual(float(frames.max()), 1)

    def test_shift_never_wraps(self):
        img = np.zeros((8, 8, 3), np.float32)
        img[1, 1, 0] = 1
        moved = shift(img, -2, -2)
        self.assertEqual(float(moved.sum()), 0)
        moved = shift(img, 3, 3)
        self.assertEqual(moved[4, 4, 0], 1)

    def test_observed_constant_velocity_moves_blob(self):
        past = np.full((3, 20, 20, 3), .025, np.float32)
        past[0, 9:12, 6:9, 0] = .95
        past[1, 9:12, 8:11, 0] = .95
        past[2, 9:12, 10:13, 0] = .95
        prediction = constant_velocity(past, 2)
        self.assertEqual(prediction.shape, (2, 20, 20, 3))
        self.assertGreater(center(prediction[0])[1], center(past[-1])[1])
        self.assertGreater(center(prediction[1])[1], center(prediction[0])[1])

    def test_baseline_cannot_observe_future_targets(self):
        seq, _ = scenes(51000, 2, 16, 20)
        a = seq.copy()
        b = seq.copy()
        b[:, 8:, ...] = 1 - b[:, 8:, ...]
        ap, av = baselines(a[:, :8], 5)
        bp, bv = baselines(b[:, :8], 5)
        np.testing.assert_array_equal(ap, bp)
        np.testing.assert_array_equal(av, bv)
        first = measure(ap, a, context=8, leads=(1, 5))
        second = measure(ap, b, context=8, leads=(1, 5))
        self.assertNotEqual(first["5"]["mse"], second["5"]["mse"])

    def test_training_smoke_writes_independent_test_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run(output=Path(folder), seeds=(3,), steps=1,
                         train_count=6, val_count=3, test_count=3,
                         batch_size=1, size=16, context=4, horizon=3,
                         ticks=1, channels=4)
            self.assertIn("single", report["summary"])
            self.assertIn("distribution", report["summary"])
            self.assertIn("persistence", report["baselines"])
            self.assertIn("constant_velocity", report["baselines"])
            self.assertTrue((Path(folder)/"seed-3-single.npz").exists())
            self.assertTrue((Path(folder)/"seed-3-distribution.npz").exists())
            self.assertTrue((Path(folder)/"metrics.json").exists())
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(report["results"][0]["seed"], 3)
            for arm in ("single", "distribution"):
                self.assertIn("3", report["results"][0]["arms"][arm]["test"])
                self.assertIn("visible_location_mae_px",
                              report["results"][0]["arms"][arm]["test"]["3"])
            self.assertIn("visible_location_mae_px",
                          report["baselines"]["constant_velocity"]["3"])


if __name__ == "__main__":
    unittest.main()
