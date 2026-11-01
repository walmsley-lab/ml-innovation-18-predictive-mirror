import tempfile
import unittest
from pathlib import Path

from experiments.rgb_motion_nca import fit


class RGBMotionNCATests(unittest.TestCase):
    def test_step5_report_contains_overfit_and_generalization_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            result = fit(seed=4, steps=12, training_steps=2, size=16,
                         context=3, horizon=2, ticks=1, channels=4,
                         generalization_seeds=(8, 9), training_seeds=(4, 5),
                         output=directory)
            self.assertEqual(result["experiment"], "rgb_motion_nca_step5")
            self.assertIn("model_mse", result["overfit"])
            self.assertIn("persistence_mse", result["overfit"])
            self.assertIn("constant_velocity_mse", result["generalization"])
            self.assertEqual(result["training_seeds"], [4, 5])
            self.assertTrue((Path(directory) / "metrics.json").exists())
            self.assertTrue((Path(directory) / "parameters.npz").exists())


if __name__ == "__main__":
    unittest.main()
