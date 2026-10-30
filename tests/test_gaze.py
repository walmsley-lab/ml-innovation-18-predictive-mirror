import unittest
import numpy as np

from experiments.gaze import WORLD, choose_gaze, compare, observe, posterior, risk


class GazeTests(unittest.TestCase):
    def test_observations_respect_field_of_view(self):
        self.assertEqual(observe(7, 6), 1)
        self.assertEqual(observe(18, 6), -1)

    def test_posterior_eliminates_inconsistent_states(self):
        prior = np.full(WORLD, 1.0 / WORLD)
        updated = posterior(prior, 6, 1)
        self.assertAlmostEqual(float(updated[7]), 1.0)
        self.assertLess(risk(updated), 1e-9)

    def test_gaze_uses_belief_not_hidden_truth(self):
        belief = np.full(WORLD, 1.0 / WORLD)
        self.assertIn(choose_gaze(belief, 6), (-3, 0, 3))
        result = compare(budget=3)
        self.assertEqual(result["greedy"]["observation_budget"], 3)
        self.assertEqual(len(result["optimized_schedule"]), 2)


if __name__ == "__main__":
    unittest.main()
