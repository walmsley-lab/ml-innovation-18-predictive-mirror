"""Causality and Bayes identity checks for the latent-memory reference."""
import unittest

import numpy as np

from experiments.latent_forward_backward import (
    TRANSITION, evaluate, filter_sequence, reverse_kernel, smooth_sequence, stationary,
)


class LatentForwardBackwardTests(unittest.TestCase):
    def test_reverse_is_stochastic_and_satisfies_bayes(self):
        pi = stationary()
        reversed_ = reverse_kernel()
        np.testing.assert_allclose(reversed_.sum(axis=1), 1, atol=1e-12)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(pi[i] * TRANSITION[i, j], pi[j] * reversed_[j, i], places=12)

    def test_future_never_changes_past_online_belief(self):
        prefix = [0, 0, 1, 1, 0]
        online = filter_sequence(prefix)
        extended = filter_sequence(prefix + [1, 1, 1])
        np.testing.assert_allclose(online, extended[:len(prefix)], atol=1e-12)

    def test_backward_information_is_retroactive_only(self):
        observations = [0, 0, 1, 1, 1, 1]
        online = filter_sequence(observations)
        smoothed = smooth_sequence(observations)
        np.testing.assert_allclose(smoothed[-1], online[-1], atol=1e-12)
        self.assertGreater(np.max(np.abs(smoothed[1] - online[1])), 1e-4)

    def test_small_seeded_run(self):
        r = evaluate(episodes=3, steps=6, seed=19)
        self.assertEqual(r["forecasts"], 9)
        self.assertEqual(set(r["nll_nats_per_forecast"]),
                         {"history", "last_observation", "shuffled_past", "unconditional"})
        self.assertTrue(all(np.isfinite(v) for v in r["nll_nats_per_forecast"].values()))


if __name__ == "__main__":
    unittest.main()
