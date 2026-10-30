import unittest
import numpy as np

from evaluate import score
from src.world import sequence


class OcclusionMetricTests(unittest.TestCase):
    def test_offscreen_reentry_is_not_barrier_occlusion(self):
        masks = np.zeros((3, 4, 4, 1), bool)
        masks[0, 1, 1, 0] = True
        masks[2, 1, 2, 0] = True
        frames = masks.astype(np.float32)
        predicted = np.zeros((2, 4, 4, 1), np.float32)
        predicted[1, 1, 2, 0] = 1.0

        offscreen = score(predicted, frames, masks, np.array([False, False, False]))
        self.assertEqual(offscreen["reappearance_count"], 0)
        self.assertIsNone(
            offscreen.get("reappearance_location_mae_px", None))
        actual_barrier = score(predicted, frames, masks,
                               np.array([False, True, False]))
        self.assertEqual(actual_barrier["reappearance_count"], 1)
        self.assertEqual(actual_barrier["reappearance_sum_abs_error_px"], 0.0)

    def test_occlusion_truth_is_consistent_with_geometry(self):
        for seed in range(15):
            frames, actions, hidden = sequence(seed, steps=12)
            self.assertEqual(len(hidden["occluded"]), len(frames))
            expected = ((hidden["world_x"] == (16 + 6) // 2)
                        & hidden["in_view"])
            np.testing.assert_array_equal(hidden["occluded"], expected)
            self.assertFalse(np.any(hidden["occluded"] &
                                    hidden["visible_mask"].any(axis=(1, 2, 3))))
            self.assertEqual(actions.shape, (12, 1))


if __name__ == "__main__":
    unittest.main()
