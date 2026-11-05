import tempfile
import unittest
import numpy as np
from src import camera_robustness as sensor
from experiments.camera_robustness import run


class CameraRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.frames=np.full((40,16,16,3),.2,np.float32)

    def test_missing_capture_metadata_does_not_invent_values(self):
        records=sensor.metadata(np.arange(3)*.1)
        self.assertIsNone(records[0]["exposure"])
        self.assertIsNone(records[1]["white_balance"])
        with self.assertRaises(ValueError):
            sensor.metadata([0.,.1,.1])

    def test_drop_rates_and_causal_imputation(self):
        for mode in ("isolated_drop","burst_drop"):
            data=sensor.corrupt(self.frames,mode,.10,seed=1)
            self.assertLessEqual(data["missing"].sum(),4)
            self.assertGreater(data["missing"].sum(),0)
            for t in np.flatnonzero(data["missing"]):
                np.testing.assert_array_equal(data["frames"][t],data["frames"][t-1])
        with self.assertRaises(ValueError):
            sensor.corrupt(self.frames,"isolated_drop",.2)

    def test_exposure_stratification_and_lead(self):
        value=sensor.corrupt(self.frames,"exposure")
        self.assertIsNotNone(value["capture"][2]["exposure"])
        strata=sensor.strata(np.ones(40),value["capture"])
        self.assertGreater(strata["exposure_changed"]["frames"],0)
        self.assertAlmostEqual(sensor.effective_lead_ms(1.,1.1,20),80)

    def test_small_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            result=run(root+"/metrics.json",seeds=(1,),context=4,
                       horizon=3,size=20,drop_rates=(.1,))
            self.assertEqual(len(result["results"]),6)


if __name__=="__main__":
    unittest.main()
