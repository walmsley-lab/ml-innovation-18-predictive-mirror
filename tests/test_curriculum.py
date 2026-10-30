import unittest

import jax
import numpy as np

from evaluate import evaluate
from src.baselines import persistence, velocity
from src.curriculum import schedule
from src.model import init
from src.world import sequence


class CurriculumTests(unittest.TestCase):
    def test_staged_and_shuffled_have_identical_example_multisets(self):
        ordered = schedule(44, 12, 5, "staged")
        shuffled = schedule(44, 12, 5, "shuffled")
        np.testing.assert_array_equal(np.sort(ordered[0].ravel()), np.sort(shuffled[0].ravel()))
        self.assertEqual(sorted(zip(ordered[0].ravel(), ordered[1].ravel())),
                         sorted(zip(shuffled[0].ravel(), shuffled[1].ravel())))
        self.assertFalse(np.array_equal(ordered[1], shuffled[1]))

    def test_motion_and_camera_crossed_conditions(self):
        for object_motion in (False, True):
            for camera in (False, True):
                frames, actions, truth = sequence(9, 10, object_motion=object_motion, camera=camera)
                self.assertEqual(frames.shape, (11, 16, 16, 1))
                if not object_motion:
                    self.assertEqual(len(set(truth["world_x"])), 1)
                if not camera:
                    self.assertTrue(np.all(actions == 0))

    def test_baselines_and_evaluator(self):
        frames, actions, truth = sequence(3, 8)
        self.assertEqual(persistence(frames).shape, (8, 16, 16, 1))
        self.assertEqual(velocity(frames, actions).shape, (8, 16, 16, 1))
        result = evaluate(init(jax.random.PRNGKey(3)), [3000], steps=8)
        self.assertEqual(set(result), {
            "still_object_still_camera", "moving_object_still_camera",
            "still_object_moving_camera", "moving_object_moving_camera"
        })


if __name__ == "__main__":
    unittest.main()
