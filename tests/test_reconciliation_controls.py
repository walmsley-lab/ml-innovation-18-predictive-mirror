import unittest

import jax
import numpy as np

from experiments.reconciliation_controls import compare, within_sheet_lesion
from src import model, multisheet
from src.world import sequence


class ReconciliationControlsTests(unittest.TestCase):
    def test_random_lesion_cuts_exactly_216_within_sheet_weights(self):
        params = multisheet.init(jax.random.PRNGKey(3))
        after = within_sheet_lesion(params, 71)
        self.assertEqual(sum(np.count_nonzero(np.asarray(params[k]) != np.asarray(after[k]))
                             for k in ("fast", "slow")), 216)
        np.testing.assert_array_equal(params["fast"][:, :, 8:, :], after["fast"][:, :, 8:, :])
        np.testing.assert_array_equal(params["slow"][:, :, 4:, :], after["slow"][:, :, 4:, :])

    def test_mean_feedback_and_correctness_controls_run(self):
        multi = multisheet.init(jax.random.PRNGKey(7))
        single = model.init(jax.random.PRNGKey(7))
        result = compare(multi, single, range(10_000, 10_008), range(20_000, 20_003))
        self.assertEqual(len(result), 5)
        self.assertIn("single_sheet_split_null", result)
        for outcome in result.values():
            self.assertIn("accurate_and_compatible_fraction", outcome)
            self.assertIn("occluded_frames", outcome)
        with self.assertRaises(ValueError):
            compare(multi, single, [10], [10])


if __name__ == "__main__":
    unittest.main()
