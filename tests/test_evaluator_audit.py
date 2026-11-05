"""Guard future-input boundaries, identifiable paired histories and known scores."""
import unittest
import numpy as np
from src import guaranteed_reappearance as world
from src.evaluator_audit import (identifiable, inspect_pair,
                                 validate_observation, audit_controls, suite)


class EvaluatorAuditTests(unittest.TestCase):
    def test_predictor_gets_only_context(self):
        rec = world.episode(11)
        self.assertTrue(validate_observation(rec, 4))
        self.assertEqual(world.model_inputs(rec)["frames"].shape[0], 4)
        self.assertFalse(np.array_equal(world.model_inputs(rec)["frames"][-1],
                                        rec["frames"][-1]))

    def test_opposing_histories_have_same_last_image_and_different_futures(self):
        for seed in (11, 23, 41):
            self.assertEqual(inspect_pair(seed)["case"], "identifiable_history")

    def test_genuinely_ambiguous_futures_not_mislabelled_as_predictable(self):
        x = np.zeros((4, 8, 8, 3))
        left, right = np.zeros((2, 8, 8, 3)), np.ones((2, 8, 8, 3))
        self.assertEqual(identifiable(x, x.copy(), left, right), "ambiguous")
        self.assertEqual(identifiable(x, x.copy(), left, left), "same_future")

    def test_oracle_positive_and_blank_negative_controls(self):
        for mode in ("barrier", "camera_exit"):
            report = audit_controls(11, mode=mode)
            self.assertTrue(report["oracle_correct"])
            self.assertTrue(report["blank_rejected"])

    def test_suite_cannot_be_mistaken_for_model_performance(self):
        result = suite((11,))
        self.assertEqual(result["learned_object_permanence"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
