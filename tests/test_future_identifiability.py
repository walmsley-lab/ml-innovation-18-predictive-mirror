"""Memory opportunity is different from irreducible future ambiguity."""
import unittest
import numpy as np
from src.future_identifiability import pair, ambiguity_bounds, audit


class IdentifiabilityTests(unittest.TestCase):
    def test_recoverable_histories_share_only_final_frame(self):
        x=pair(11,"history_identifiable")
        self.assertFalse(x["identical_full_context"])
        self.assertFalse(x["identical_future"])
        np.testing.assert_array_equal(x["contexts"][0,-1],x["contexts"][1,-1])
        self.assertGreater(float(np.mean(abs(x["futures"][0]-x["futures"][1]))),0)

    def test_hidden_stochastic_turn_is_unidentifiable_from_full_observation(self):
        x=pair(11,"unobserved_turn_ambiguous")
        self.assertTrue(x["identical_full_context"])
        self.assertFalse(x["identical_future"])
        np.testing.assert_array_equal(x["contexts"][0],x["contexts"][1])
        bound=ambiguity_bounds(x["futures"])
        self.assertGreater(bound["conditional_mean_mse"],0)
        self.assertGreater(bound["single_branch_guess_mse"],
                           bound["conditional_mean_mse"])
        self.assertEqual(bound["oracle_privileged_mse"],0)

    def test_same_future_null_has_zero_irreducible_point_error(self):
        x=pair(23,"same_future_null")
        self.assertTrue(x["identical_full_context"])
        self.assertTrue(x["identical_future"])
        self.assertEqual(ambiguity_bounds(x["futures"])["conditional_mean_mse"],0)

    def test_audit_has_no_trained_model_claims(self):
        report=audit((11,23))
        self.assertEqual(len(report["records"]),6)
        self.assertIn("no learned model",report["purpose"])


if __name__=="__main__":
    unittest.main()
