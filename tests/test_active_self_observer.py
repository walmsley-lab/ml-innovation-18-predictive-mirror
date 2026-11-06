"""Tests for action-before-observation and memory-ablated foveal comparison."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.active_self_observer import (
    choose_gaze, episode, foveate, initial_belief, observations,
    red_center, run,
)


class ActiveSelfObserverTests(unittest.TestCase):
    def test_one_aperture_and_retinal_detector(self):
        image = np.full((16, 16, 3), .035, np.float32)
        image[4, 6] = (.95, .2, .12)
        crop, mask = foveate(image, (6, 4), 3)
        self.assertLess(mask.sum(), image.shape[0] * image.shape[1])
        np.testing.assert_allclose(crop[0, 0], [.035] * 3)
        xy = red_center(crop)
        self.assertIsNotNone(xy)
        np.testing.assert_allclose(xy, [6, 4], atol=.01)

    def test_policy_uses_prior_belief_only(self):
        state = initial_belief(32)
        first = choose_gaze(state, 2, "active", 32, 6)
        state["position"] += (3, -2)
        shifted = choose_gaze(state, 2, "active", 32, 6)
        self.assertFalse(np.array_equal(first, shifted))
        np.testing.assert_allclose(choose_gaze(state, 2, "center", 32, 6),
                                   [16, 16])

    def test_heldout_truth_and_occlusion_are_evaluator_only(self):
        seen, truth, occluded = observations(4011, 32, 12)
        self.assertTrue(occluded.any())
        self.assertIsNotNone(red_center(truth[np.flatnonzero(occluded)[0]]))
        self.assertIsNone(red_center(seen[np.flatnonzero(occluded)[0]]))
        result = episode(seen, truth, occluded, 11, "active", 6, 8)
        self.assertEqual(len(result["trace"]), 36)
        sample = result["trace"][1]
        self.assertIn("true_xy", sample["evaluation"])
        self.assertNotIn("true_xy", sample["observer"])
        self.assertTrue(np.isfinite(result["mean_tracking_error_px"]))

    def test_comparison_executes_and_lesion_changes_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run(tmp, seeds=(11, 23), size=32,
                         segment_frames=12, radius=6, sheet_size=8)
            self.assertEqual(len(report["results"]), 2)
            for entry in report["results"]:
                self.assertEqual(set(entry["arms"]), {
                    "active", "center", "raster", "active_no_memory"})
                for arm in entry["arms"].values():
                    self.assertTrue(np.isfinite(arm["mean_tracking_error_px"]))
                    self.assertTrue((Path(tmp) / arm["trace_file"]).is_file())
                    records = [json.loads(line) for line in
                               (Path(tmp) / arm["trace_file"]).read_text().splitlines()]
                    self.assertEqual(len(records), 36)
                    self.assertEqual(
                        len({r["evaluation"]["aperture_pixels"] for r in records}) >= 1,
                        True)
            def gaze(name):
                return [json.loads(line)["observer"]["gaze_xy"] for line in
                        (Path(tmp) / entry["arms"][name]["trace_file"]).read_text().splitlines()]
            entry = report["results"][0]
            self.assertNotEqual(gaze("active"), gaze("active_no_memory"))
            self.assertTrue((Path(tmp) / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
