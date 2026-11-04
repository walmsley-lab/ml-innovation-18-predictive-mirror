"""Verify the event evaluator on known cases BEFORE trusting any model score.

Four controlled forecasts with a known answer exercise all three metric views
(hard threshold detector, foreground segmentation, soft threshold-free
localization):

* perfect prediction      -> zero error, full overlap, mass fraction 1, no miss
* shifted object          -> localization error ~= the shift, in both views
* missing object (black)  -> hard miss + zero recall + ~zero soft mass
                             (a genuine disappearance)
* dim / below-threshold    -> hard detector misses, soft view keeps the object
                             (a detection-threshold artifact, not disappearance)
* false detection         -> a disc where none is predicted hurts precision
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.rgb_event_benchmark import run
from src import rgb_event_metrics as metrics
from src.rgb_event_metrics import (score, soft_localization,
                                   foreground_segmentation, object_saliency)
from src.rgb_motion_world import COLORS, episode


class RGBEventMetricsTests(unittest.TestCase):
    def setUp(self):
        self.records = [episode(123, steps=9, size=24, balls=2, occlusion=True)]
        self.truth = self.records[0]["frames"][4:9][None]

    # ---- known-case verification of the primitives -------------------------
    def test_soft_localization_recovers_a_known_disc(self):
        size = 24
        image = np.zeros((size, size, 3), np.float32)
        image[10:14, 6:10] = COLORS[1]  # a cyan square centred ~(11.5, 7.5)
        mass, centroid = soft_localization(image, COLORS[1])
        self.assertGreater(mass, 0)
        np.testing.assert_allclose(centroid, [11.5, 7.5], atol=0.5)
        # Empty frame -> no mass, undefined centroid.
        empty_mass, empty_centroid = soft_localization(
            np.zeros_like(image), COLORS[1])
        self.assertLess(empty_mass, 1e-6)
        self.assertIsNone(empty_centroid)
        # A different colour contributes almost no mass to this object.
        other = np.zeros_like(image)
        other[10:14, 6:10] = COLORS[0]
        self.assertLess(soft_localization(other, COLORS[1])[0], 0.2 * mass)

    def test_segmentation_perfect_and_disjoint(self):
        target = np.zeros((16, 16, 3), np.float32)
        target[4:8, 4:8] = COLORS[0]
        perfect = foreground_segmentation(target.copy(), target)
        self.assertAlmostEqual(perfect["iou"], 1.0)
        self.assertAlmostEqual(perfect["recall"], 1.0)
        self.assertAlmostEqual(perfect["precision"], 1.0)
        disjoint_pred = np.zeros_like(target)
        disjoint_pred[10:14, 10:14] = COLORS[0]
        disjoint = foreground_segmentation(disjoint_pred, target)
        self.assertEqual(disjoint["iou"], 0.0)
        self.assertEqual(disjoint["recall"], 0.0)
        empty = foreground_segmentation(np.zeros_like(target), target)
        self.assertEqual(empty["recall"], 0.0)
        self.assertIsNone(empty["precision"])  # nothing predicted

    # ---- known-case verification through score() ---------------------------
    def test_perfect_forecast_scores_zero(self):
        result = score(self.truth, self.records, context=4, leads=(1, 3, 5))
        for values in result.values():
            self.assertAlmostEqual(values["mse"], 0)
            self.assertEqual(values["missing_visible_predictions"], 0)
            if values["foreground_mse"] is not None:
                self.assertAlmostEqual(values["foreground_mse"], 0)
            if values["visible_location_mae_px"] is not None:
                self.assertAlmostEqual(values["visible_location_mae_px"], 0)
            if values["soft_centroid_mae_px"] is not None:
                self.assertLess(values["soft_centroid_mae_px"], 0.5)
            if values["foreground_iou"] is not None:
                self.assertAlmostEqual(values["foreground_iou"], 1.0)
            if values["predicted_mass_fraction"] is not None:
                self.assertGreater(values["predicted_mass_fraction"], 0.9)

    def test_shifted_forecast_reports_the_shift(self):
        shift = 2
        shifted = np.roll(self.truth, shift, axis=3)  # move columns by +2 px
        result = score(shifted, self.records, context=4, leads=(1,))["1"]
        # Both localization views should see roughly the induced shift.
        self.assertGreater(result["visible_location_mae_px"], 0.5)
        if result["soft_centroid_mae_px"] is not None:
            self.assertGreater(result["soft_centroid_mae_px"], 0.5)
            self.assertLess(result["soft_centroid_mae_px"], 2 * shift + 2)
        # The object is still present: mass preserved, nothing "missing".
        self.assertGreater(result["predicted_mass_fraction"], 0.5)

    def test_missing_foreground_is_a_genuine_disappearance(self):
        missing = np.zeros_like(self.truth)
        result = score(missing, self.records, 4, (1, 3, 5))
        visible_count = sum(v["visible_objects"] for v in result.values())
        missed = sum(v["missing_visible_predictions"] for v in result.values())
        self.assertGreater(visible_count, 0)
        self.assertEqual(missed, visible_count)  # hard detector: all missed
        for values in result.values():
            self.assertGreater(values["mse"], 0)
            # Soft view agrees it is a real disappearance, not a threshold issue.
            self.assertLess(values["predicted_mass_fraction"], 0.05)
            self.assertEqual(values["soft_localized_objects"], 0)
            self.assertEqual(values["foreground_recall"], 0.0)

    def test_dim_object_is_a_threshold_artifact_not_disappearance(self):
        # Globally dim the frame: the hue is preserved but both the brightness
        # gate (0.30) and the colour-distance gate of the hard detector fail,
        # so it reports every object "missing" -- yet the object is really there.
        very_dim = self.truth * 0.25  # 0.95*0.25 = 0.2375 < 0.30 gate
        hard = score(very_dim, self.records, 4, (1,))["1"]
        self.assertEqual(hard["missing_visible_predictions"], hard["visible_objects"])
        # The intensity-robust soft view keeps the object: appreciable mass,
        # localized close to the truth. This is a detection-threshold artifact,
        # not a disappearance.
        self.assertGreater(hard["predicted_mass_fraction"], 0.5)
        self.assertGreater(hard["soft_localized_objects"], 0)
        self.assertIsNotNone(hard["soft_centroid_mae_px"])
        self.assertLess(hard["soft_centroid_mae_px"], 2.0)

    def test_false_detection_hurts_precision(self):
        # Start from truth, paint an extra off-object disc of object-0 colour.
        polluted = self.truth.copy()
        polluted[0, 0, 0:3, 0:3] = COLORS[0]  # bright blob in a corner at lead 1
        clean = foreground_segmentation(self.truth[0, 0], self.records[0]["frames"][4])
        dirty = foreground_segmentation(polluted[0, 0], self.records[0]["frames"][4])
        # Extra predicted foreground with no truth support lowers precision.
        self.assertGreater(dirty["predicted_foreground_pixels"],
                           clean["predicted_foreground_pixels"])
        if clean["precision"] is not None and dirty["precision"] is not None:
            self.assertLessEqual(dirty["precision"], clean["precision"])

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            score(self.truth, self.records, 4, (6,))

    def test_runner_outputs_baselines_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "metrics.json"
            report = run(output=out, episodes=2, context=4, horizon=3,
                         size=24, balls=1, occlusion=False)
            self.assertTrue(out.exists())
            self.assertEqual(set(report["metrics"]),
                             {"persistence", "constant_velocity"})
            row = report["metrics"]["persistence"]["3"]
            for key in ("foreground_mse", "foreground_iou",
                        "predicted_mass_fraction", "soft_centroid_mae_px"):
                self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
