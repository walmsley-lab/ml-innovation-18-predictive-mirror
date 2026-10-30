import unittest

import numpy as np

from experiments.plasticity import compare, simulate, update


class PlasticityTests(unittest.TestCase):
    def test_fixed_weights_really_are_fixed(self):
        activity = np.ones(12) * 0.5
        weights = np.ones((12, 2)) * 0.15
        _, returned = update(activity, weights, np.zeros(12), "fixed")
        np.testing.assert_array_equal(returned, weights)

    def test_homeostasis_limits_unbounded_hebbian_growth_in_this_toy(self):
        results = compare(seeds=(4, 7), steps=120)
        self.assertGreater(results["hebbian"]["mean_final_max_weight"],
                           results["homeostatic"]["mean_final_max_weight"])
        self.assertAlmostEqual(results["fixed"]["mean_final_max_weight"], 0.15)

    def test_reproducibility_and_unknown_rules(self):
        self.assertEqual(simulate(12, steps=8), simulate(12, steps=8))
        with self.assertRaises(ValueError):
            update(np.zeros(3), np.ones((3, 2)), np.zeros(3), "magic")


if __name__ == "__main__":
    unittest.main()
