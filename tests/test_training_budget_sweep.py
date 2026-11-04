"""Mechanics + labelling checks for the training-budget sweep."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.training_budget_sweep import (run, classify, ARMS,
                                               foreground_episodes)
from src import rgb_motion_world


class TrainingBudgetSweepTests(unittest.TestCase):
    def test_classify_labels(self):
        collapsed = {"clip_rail_saturation": 1.0, "input_independence": 0.0,
                     "predicted_mass_fraction": 0.0, "foreground_recall": 0.0}
        learning = {"clip_rail_saturation": 0.2, "input_independence": 0.03,
                    "predicted_mass_fraction": 0.6, "foreground_recall": 0.4}
        under = {"clip_rail_saturation": 0.5, "input_independence": 0.02,
                 "predicted_mass_fraction": 0.05, "foreground_recall": 0.05}
        self.assertEqual(classify(collapsed), "collapsed")
        self.assertEqual(classify(learning), "learning")
        self.assertEqual(classify(under), "undertrained")

    def test_foreground_episodes_nonempty(self):
        records = [rgb_motion_world.episode(31000 + i, steps=12, size=24,
                                            balls=2, occlusion=True)
                   for i in range(6)]
        pool = foreground_episodes(records, context=6)
        self.assertTrue(pool)
        self.assertTrue(all(0 <= i < len(records) for i in pool))

    def test_smoke_runs_all_arm_kinds_and_saves_checkpoints(self):
        arms = ("cnn", "transport", "nca", "routing", "hierarchical",
                "nca_blend", "nca_grounded")
        with tempfile.TemporaryDirectory() as tmp:
            report = run(out=tmp, arms=arms, seeds=(11,), budgets=(1, 2),
                         context=4, horizon=3, size=24, width=4,
                         batch_size=2, train_count=6, val_count=3)
            self.assertEqual(len(report["results"]), len(arms))
            for result in report["results"]:
                self.assertEqual(set(result["budgets"]), {"1", "2"})
                for budget in result["budgets"].values():
                    for key in ("mse", "foreground_recall",
                                "predicted_mass_fraction", "clip_rail_saturation",
                                "status"):
                        self.assertIn(key, budget)
            # Per-budget checkpoints exist.
            self.assertTrue((Path(tmp) / "nca_grounded-seed11-step2.npz").exists())
            self.assertTrue((Path(tmp) / "cnn-seed11-step1.npz").exists())
            self.assertTrue((Path(tmp) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
