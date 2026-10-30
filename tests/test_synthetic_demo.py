import unittest

import numpy as np

from synthetic_demo import event_metrics


class SyntheticDemoTests(unittest.TestCase):
    def test_event_metrics_reports_detection_and_miss(self):
        frames = np.zeros((4, 4, 4, 1), np.float32)
        visible = np.zeros_like(frames, dtype=bool)
        visible[0, 1, 1, 0] = True
        visible[3, 1, 2, 0] = True
        truth = {"visible_mask": visible, "occluded": np.array([False, True, True, False])}

        detected = frames[:-1].copy()
        detected[2, 1, 2, 0] = 1.0
        metrics = event_metrics(detected, frames, truth)
        self.assertEqual(metrics["event_count"], 1)
        self.assertEqual(metrics["detected_event_count"], 1)
        self.assertEqual(metrics["detection_rate"], 1.0)
        self.assertEqual(metrics["localization_mae_px"], 0.0)

        missed = np.zeros_like(detected)
        metrics = event_metrics(missed, frames, truth)
        self.assertEqual(metrics["detected_event_count"], 0)
        self.assertEqual(metrics["detection_rate"], 0.0)
        self.assertEqual(metrics["localization_mae_px"], 4.0)


if __name__ == "__main__":
    unittest.main()
