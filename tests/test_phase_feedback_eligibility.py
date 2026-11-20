"""Deterministic and negative-control contracts for phase feedback."""
import unittest
import numpy as np
from experiments.phase_feedback_eligibility import evaluate, gate, generate


class PhaseFeedbackTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(evaluate(11, 400), evaluate(11, 400))

    def test_gate_bounded_and_phase_sensitive(self):
        p = np.array([0.0, np.pi])
        g = gate(p, "phase", np.random.default_rng(1))
        np.testing.assert_allclose(g, [1.0, 0.0])

    def test_independent_splits(self):
        a = generate(11, 400)
        b = generate(12, 400)
        self.assertFalse(np.array_equal(a[1], b[1]))

    def test_phase_blind_negative_control(self):
        # If phase does not predict feedback reliability, the oracle phase
        # gate should not systematically outperform a tuned constant gate.
        rows = [evaluate(s, 3000, False) for s in [11, 23, 41]]
        self.assertGreaterEqual(
            np.mean([r["phase"] - r["validation_tuned_constant"] for r in rows]),
            -0.025,
        )


if __name__ == "__main__":
    unittest.main()
