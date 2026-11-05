import unittest
import numpy as np
from src import identity_uncertainty_world as world


class IdentityTests(unittest.TestCase):
    def test_hidden_identity_swap_is_unobservable_in_rgb(self):
        a,b=world.paired_hidden_identity(11)
        np.testing.assert_array_equal(a["frames"],b["frames"])
        self.assertFalse(a["identity_observable_from_rgb"])
        self.assertIsNone(world.identity_score(0,a))
        self.assertGreater(float(np.abs(
            a["positions"][8:]-b["positions"][8:]).mean()),0)

    def test_appearance_identity_is_defined_when_distinct(self):
        a=world.episode(11,appearance="distinct")
        self.assertEqual(world.identity_score(0,a),1)
        self.assertEqual(world.identity_score(1,a),0)

    def test_uncertainty_scores(self):
        self.assertAlmostEqual(world.branch_brier(.5,True),.25)
        self.assertAlmostEqual(world.branch_brier(.5,False),.25)
        with self.assertRaises(ValueError):
            world.branch_log_loss(1.1,True)


if __name__=="__main__":
    unittest.main()
