import unittest

import numpy as np

from src.rgb_motion_world import batch, episode


class RGBMotionWorldTests(unittest.TestCase):
    def test_deterministic_rgb_frames_and_truth(self):
        first = episode(7, steps=12, size=24, balls=2)
        second = episode(7, steps=12, size=24, balls=2)
        np.testing.assert_array_equal(first["frames"], second["frames"])
        np.testing.assert_array_equal(first["positions"], second["positions"])
        self.assertEqual(first["frames"].shape, (12, 24, 24, 3))
        self.assertEqual(first["positions"].shape, (12, 2, 2))
        self.assertGreater(float(np.abs(first["frames"][1:] - first["frames"][:-1]).sum()), 0)
        self.assertGreater(float(np.max(first["frames"])), 0.5)

    def test_world_truth_is_separate_from_training_batch(self):
        data = episode(11, steps=10, size=24, balls=1, occlusion=True,
                       camera_motion=True)
        training = batch([11, 12], steps=10, size=24, balls=1,
                         occlusion=True, camera_motion=True)
        self.assertEqual(training.shape, (2, 10, 24, 24, 3))
        self.assertTrue(np.isfinite(training).all())
        self.assertGreater(float(np.abs(data["frames"][:, :, 10:14] -
                                         data["frames"][:, :, 0:4]).sum()), 0)
        self.assertEqual(data["visible"].dtype, np.bool_)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            episode(1, steps=1)
        with self.assertRaises(ValueError):
            episode(1, balls=0)


if __name__ == "__main__":
    unittest.main()
