import unittest
import numpy as np
from experiments import roadmap_31_rgb_calibration as m


class RGBIntervalTests(unittest.TestCase):
    def test_context_only_forecast(self):
        batch=m.frames(61000,10,context=3,horizon=2,size=16)
        pred=m.forecast(batch[:,:3],2,"constant_velocity")
        edited=batch.copy()
        edited[:,3:]=1-edited[:,3:]
        np.testing.assert_allclose(pred,m.forecast(edited[:,:3],2,
                                                  "constant_velocity"))
        self.assertEqual(pred.shape,(10,2,16,16,3))
        with self.assertRaises(ValueError):
            m.forecast(batch[:,:3],2,"nca")

    def test_calibration_and_test_seeds_are_independent(self):
        calibration=m.frames(61000,10,3,2,16)
        test=m.frames(91000,10,3,2,16)
        self.assertFalse(np.array_equal(calibration,test))
        report=m.run(None,calibration_count=12,test_count=12,
                     context=3,horizon=2,size=16)
        self.assertIn("all_pixels_per_lead_coverage",
                      report["test"]["conformal"])
        self.assertEqual(len(report["calibrated_radius_per_lead"]),2)

    def test_score_shape_guard(self):
        batch=m.frames(61000,10,3,2,16)
        with self.assertRaises(ValueError):
            m.abs_max_errors(np.zeros((10,1,16,16,3)),batch,3)


if __name__=="__main__":
    unittest.main()
