import tempfile
import unittest
import numpy as np
from src import field_perturbations as probe
from experiments.field_perturbation_probe import run


class FieldPerturbationTests(unittest.TestCase):
    def test_calibrated_diffusion_and_wave(self):
        checks=probe.calibrate(size=16,steps=10)
        self.assertLess(checks["heat_impulse_mass_max_error"],1e-8)
        self.assertGreater(checks["diffusion_final_radius2"],
                           checks["diffusion_start_radius2"])
        self.assertGreater(checks["wave_displacement_response_max"],0)
        self.assertEqual(checks["zero_control_l2"],0)

    def test_matched_energy_shift(self):
        controls=probe.match_controls(size=16,steps=8)
        self.assertAlmostEqual(np.linalg.norm(controls["center"]["response"][0]),
                               np.linalg.norm(controls["shifted"]["response"][0]))
        self.assertTrue(np.array_equal(
            controls["zero"],np.zeros_like(controls["zero"])))

    def test_untrained_free_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            r=run(folder+"/metrics.json",seeds=(1,),steps=8,size=16)
            self.assertFalse(r["checkpoint_evaluated"])
            self.assertGreater(r["rows"][0]["positive_control_l2"][0],0)


if __name__=="__main__":
    unittest.main()
