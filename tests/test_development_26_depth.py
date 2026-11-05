import unittest
import numpy as np
from experiments import development_26_depth as exp


class Development26Tests(unittest.TestCase):
    def test_zero_padding_and_geometry(self):
        x=np.zeros((4,8)); x[1,1]=1
        y=exp.shift(x,2)
        self.assertEqual(y[1,3],1)
        self.assertEqual(y[1,1],0)
        self.assertEqual(exp.shift(x,-10).sum(),0)

    def test_known_disparities_and_depths(self):
        for seed in (11,12,13,91000):
            left,right,z,disp=exp.pair(seed)
            inferred_z,inferred_d=exp.estimate(left,right)
            np.testing.assert_array_equal(inferred_d,disp)
            np.testing.assert_allclose(inferred_z,z,atol=1e-6)

    def test_invalid_calibration(self):
        left,right,_,_=exp.pair()
        with self.assertRaises(ValueError):
            exp.estimate(left,right,0)
        self.assertIn("shuffled_right", exp.run(out=None,seeds=(11,12))["depth_mae"])


if __name__=="__main__":
    unittest.main()
