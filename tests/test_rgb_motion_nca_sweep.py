import tempfile
import unittest

from experiments.rgb_motion_nca_sweep import run


class RGBMotionNCASweepTests(unittest.TestCase):
    def test_aggregate_reports_paired_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(seeds=(3,), generalization_seeds=(8,), output=directory,
                         steps=8, training_steps=1, size=16, context=3,
                         horizon=2, ticks=1, channels=4)
            self.assertEqual(result["runs"], 1)
            self.assertIn("model_mse", result["generalization_mean_mse"])
            self.assertIn("constant_velocity_mse", result["generalization_mean_mse"])
            self.assertIn(result["generalization_beats_persistence_count"], (0, 1))


if __name__ == "__main__":
    unittest.main()
