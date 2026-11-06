"""Causal controls and deterministic smoke tests for plasticity modulation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import _pool
from experiments.regulated_plasticity import (
    initial_regulator, play, regulate, run, world,
)


class RegulatedPlasticityTests(unittest.TestCase):
    def test_regulator_explores_then_consolidates(self):
        state = initial_regulator()
        # Large untrained forecast errors must not poison the baseline.
        for error in (.009, .009, .0003):
            state, rate = regulate(state, error, .04)
            self.assertEqual(rate, .04)
        for _ in range(2):
            state, rate = regulate(state, .001, .04)
        self.assertEqual(state["mode"], "explore")
        self.assertGreater(rate, .04)
        for _ in range(8):
            state, _ = regulate(state, .001, .04)
        self.assertEqual(state["mode"], "consolidate")
        state, rate = regulate(state, .001, .04)
        self.assertLess(rate, .04)
        with self.assertRaises(ValueError):
            regulate(state, float("nan"), .04)

    def test_motion_change_actuates_gate(self):
        # Mechanism check on the diagnostic seed. This is not held-out
        # evidence that the adaptive model predicts any better.
        frames = world(4011, size=32, segment_frames=40)
        adaptive = play(frames, seed=11, sheet_size=8)
        explore = [i for i, mode in enumerate(adaptive["modes"])
                   if mode == "explore"]
        self.assertTrue(explore, "Changed world must exercise adaptive gate")
        self.assertGreaterEqual(explore[0], 40)
        self.assertLess(explore[0], 80)
        self.assertIn("consolidate", adaptive["modes"])
        self.assertGreater(adaptive["rates"].max(), .04)
        self.assertLess(adaptive["rates"].min(), .04)
        fixed = play(frames, seed=11, sheet_size=8,
                     schedule=np.full(len(frames), .04))
        self.assertGreater(np.max(np.abs(adaptive["final_weights"] -
                                         fixed["final_weights"])), 0.)

    def test_prequential_score_and_replay(self):
        frames = world(12, 16, 8)
        adaptive = play(frames, 12, sheet_size=4)
        expected = float(np.mean(_pool(frames[0], 4) ** 2))
        self.assertAlmostEqual(adaptive["errors"][0], expected, places=7)
        replay = play(frames, 12, sheet_size=4, schedule=adaptive["rates"])
        np.testing.assert_allclose(replay["errors"], adaptive["errors"], atol=1e-7)
        np.testing.assert_allclose(replay["final_weights"], adaptive["final_weights"], atol=1e-7)

    def test_paired_smoke_and_budget_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run(folder, seeds=(11, 23), segment_frames=8, size=16,
                         sheet_size=4)
            self.assertEqual(len(report["results"]), 2)
            for seed in report["results"]:
                arms = seed["arms"]
                self.assertEqual(set(arms), {
                    "adaptive", "fixed", "matched_constant", "shifted_schedule"})
                adaptive = arms["adaptive"]
                for name in ("matched_constant", "shifted_schedule"):
                    self.assertAlmostEqual(
                        arms[name]["total_commanded_rate"],
                        adaptive["total_commanded_rate"], places=6)
                for arm in arms.values():
                    self.assertEqual(len(arm["error_curve"]), 24)
                    self.assertTrue(np.isfinite(arm["error_curve"]).all())
            self.assertTrue(Path(folder, "metrics.json").exists())
            self.assertTrue(Path(folder, "seed-11-adaptive.npz").exists())


if __name__ == "__main__":
    unittest.main()
