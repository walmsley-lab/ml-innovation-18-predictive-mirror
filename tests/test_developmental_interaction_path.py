import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import DevelopmentalSheet, run


class DevelopmentalInteractionPathTests(unittest.TestCase):
    def test_online_connections_change_from_experience(self):
        learner = DevelopmentalSheet(sheet=4, seed=2, plasticity=.1)
        frame = np.zeros((16, 16, 3), np.float32)
        frame[4:8, 4:8] = 1
        before = learner.connections.copy()
        learner.observe(frame)
        self.assertFalse(np.array_equal(before, learner.connections))
        self.assertTrue(np.isfinite(learner.connections).all())
        self.assertTrue(np.isfinite(learner.homeostasis).all())

    def test_world3d_path_preserves_retinal_only_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(folder, frames=5, size=16, sheet=4, source="world3d")
            self.assertEqual(result["retinal_size"], [16, 16, 3])
            self.assertFalse(result["provenance"]["depth_seen_by_model"])
            self.assertTrue(Path(folder, "metrics.json").exists())
            self.assertTrue(np.isfinite(result["mean_prediction_mse"]))


if __name__ == "__main__":
    unittest.main()
