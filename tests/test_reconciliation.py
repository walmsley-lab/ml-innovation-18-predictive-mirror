import unittest

import jax
import numpy as np

from experiments.reconciliation import decode, fit_probe, observations, test_reconciliation
from src.multisheet import init


class ReconciliationTests(unittest.TestCase):
    def test_ridge_decoder_can_recover_known_linear_map(self):
        x = np.arange(25)[:, None].astype(float)
        y = 3 * x[:, 0] + 7
        w = fit_probe(x, y, ridge=0.000001)
        self.assertLess(float(np.mean(np.abs(decode(x, w) - y))), 1e-4)

    def test_probes_do_not_train_perceptual_network(self):
        params = init(jax.random.PRNGKey(8))
        x = observations(params, range(10_000, 10_003))
        self.assertEqual(x[0].shape[0], 36)
        self.assertEqual(x[0].shape[1], 16 * 6)
        self.assertEqual(x[1].shape[1], 16 * 2)
        result = test_reconciliation(params, range(10_000, 10_008), range(20_000, 20_005))
        self.assertIn("accurate_and_compatible_fraction", result)
        self.assertIn("feedback_cut_accurate_and_compatible_fraction", result)


if __name__ == "__main__":
    unittest.main()
