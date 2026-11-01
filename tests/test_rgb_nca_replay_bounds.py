"""Regression for the last valid RGB NCA held-out forecast anchor.

A 600-frame recording formerly crashed while trying to read frames[600]
at the final +5-frame forecast (the last valid index is 599).
"""
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.rgb_nca_viewer import replay


class RGBNCAReplayBoundsTests(unittest.TestCase):
    def setUp(self):
        # Distinct values make it easy to spot accidentally shifted horizons.
        values = np.arange(600, dtype=np.float32) / 600
        self.frames = np.broadcast_to(
            values[:, None, None, None], (600, 4, 4, 3)).copy()

    @staticmethod
    def predict(context):
        # Deterministic persistence forecasts, no model compilation or UI.
        anchor = np.asarray(context)[0, -1]
        forecasts = np.repeat(anchor[None, None], 5, axis=1)
        activity = np.zeros((1, 5, *anchor.shape[:2], 1), np.float32)
        return forecasts, activity

    @staticmethod
    def arguments(stride):
        return SimpleNamespace(context=6, horizon=5, leads=(1, 3, 5),
                               stride=stride, no_display=True,
                               max_frames=0, fps=10, panel_size=80)

    def test_even_stride_does_not_read_past_600th_frame(self):
        scores, latencies, count = replay(
            self.frames, self.predict, self.arguments(2), cv2=None)
        # Context starts at index 485; legal anchors end at 594.
        self.assertEqual(count, 55)
        self.assertEqual(len(latencies), count)
        for lead in (1, 3, 5):
            self.assertEqual(len(scores[str(lead)]), count)

    def test_unit_stride_includes_final_legal_anchor(self):
        scores, _, count = replay(
            self.frames, self.predict, self.arguments(1), cv2=None)
        # Inclusive valid anchors 485...594, each with +5 truth <= 599.
        self.assertEqual(count, 110)
        self.assertEqual(len(scores["5"]), 110)

    def test_short_heldout_recording_explains_failure(self):
        with self.assertRaisesRegex(ValueError, "Not enough frames"):
            replay(self.frames[:20], self.predict, self.arguments(1), cv2=None)


if __name__ == "__main__":
    unittest.main()
