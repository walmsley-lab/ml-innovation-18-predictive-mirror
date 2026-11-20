"""Contract checks for trained phase-conditioned readout."""
import unittest
import numpy as np
from experiments.learned_phase_feedback import sequences, features, fit, run_seed


class LearnedPhaseFeedbackTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(run_seed(11, 30, 12), run_seed(11, 30, 12))

    def test_equal_parameter_count(self):
        d = sequences(11, 30, 12)
        self.assertEqual(features(d, "phase").shape, features(d, "nonphase").shape)
        self.assertEqual(len(fit(d, "phase", 1)), len(fit(d, "nonphase", 1)))

    def test_future_sensory_does_not_change_past_state(self):
        d = sequences(11, 30, 12)
        altered = dict(d)
        altered["sensory"] = d["sensory"].copy()
        altered["sensory"][:, 8:] += 100
        # The state is built causally from the sensory stream in sequences().
        state = np.zeros_like(altered["sensory"])
        for t in range(12):
            state[:, t] = (0.8 * state[:, t - 1] if t else 0) + 0.2 * altered["sensory"][:, t]
        np.testing.assert_allclose(state[:, :8], d["state"][:, :8])

    def test_finite_results(self):
        for informative in (True, False):
            r = run_seed(11, 30, 12, informative)
            self.assertTrue(all(np.isfinite(v) for v in r.values()))


if __name__ == "__main__":
    unittest.main()
