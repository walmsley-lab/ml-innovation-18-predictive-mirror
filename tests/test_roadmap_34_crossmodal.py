import unittest
import numpy as np
from experiments import roadmap_34_crossmodal as m


class CrossModalTests(unittest.TestCase):
    def test_causal_sensor_feature_alignment(self):
        episode=m.episode(11)
        x=m.features(episode["frames"],episode["imu"],4)
        self.assertAlmostEqual(x[2],episode["imu"][4])
        changed={k:(v.copy() if isinstance(v,np.ndarray) else v)
                 for k,v in episode.items()}
        changed["frames"][5:]=0.
        changed["imu"][5:]=999.
        np.testing.assert_array_equal(x,m.features(changed["frames"],changed["imu"],4))

    def test_missingness_and_control_shapes(self):
        x,y,_=m.records(31000,2,steps=7)
        mask=np.arange(len(x))%2==0
        robust=m.mask_features(x,"robust",mask)
        np.testing.assert_array_equal(robust[mask,2],np.zeros(mask.sum()))
        np.testing.assert_array_equal(robust[:,3],mask.astype(float))
        for mode in ("vision","inertial","fused"):
            self.assertEqual(m.mask_features(x,mode).shape,x.shape)

    def test_independent_heldout(self):
        r=m.run(None,train_count=2,val_count=2,test_count=2)
        self.assertIn("inertial_shuffled",r["test"])
        self.assertIn("sensor_dropped",r["test"])
        self.assertIn("robust",r["test"]["sensor_dropped"])


if __name__=="__main__":
    unittest.main()
