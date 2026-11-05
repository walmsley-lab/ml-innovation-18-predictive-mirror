"""Known true forecasts and deliberately broken forecasts calibrate flags."""
import unittest
import numpy as np

from src import rgb_motion_world
from src import forecast_diagnostics as diag


class PerceptualDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.records=[rgb_motion_world.episode(
            seed,steps=8,size=32,balls=1,occlusion=False)
            for seed in (42,67,91)]
        self.context=4
        self.oracle=np.stack([r["frames"][self.context:] for r in self.records])

    def flags(self,forecast):
        report=diag.perceptual_failure_report(
            forecast,self.records,self.context,(1,))
        return report["per_lead"]["1"]

    def test_oracle_is_accurate(self):
        report=self.flags(self.oracle)
        self.assertAlmostEqual(report["pixel_mse"],0)
        self.assertIn("accurate",report["flags"])

    def test_black_bright_and_diffuse(self):
        black=self.flags(np.zeros_like(self.oracle))
        self.assertIn("black",black["flags"])
        self.assertIn("input_independent",black["flags"])
        white=self.flags(np.ones_like(self.oracle))
        self.assertIn("bright",white["flags"])
        self.assertIn("diffuse",white["flags"])
        red=self.flags(np.broadcast_to(
            np.array([.95,.18,.12],np.float32),self.oracle.shape).copy())
        self.assertIn("diffuse",red["flags"])

    def test_shifted_is_misplaced(self):
        shifted=np.roll(self.oracle,6,axis=3)
        self.assertIn("misplaced",self.flags(shifted)["flags"])

    def test_single_scene_independence_not_claimed(self):
        report=diag.perceptual_failure_report(
            self.oracle[:1],self.records[:1],self.context,(1,))
        row=report["per_lead"]["1"]
        self.assertFalse(row["input_independence_testable"])
        self.assertNotIn("input_independent",row["flags"])


if __name__=="__main__":
    unittest.main()
