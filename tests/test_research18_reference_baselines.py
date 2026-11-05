import unittest
import numpy as np
from src import research18_reference_baselines as b
class Tests(unittest.TestCase):
    def test_persistence(self):
        np.testing.assert_array_equal(b.persistence([1,2,3]),[1,1,2])
        np.testing.assert_array_equal(b.delayed([1,2,3],2)[2:],[1])
    def test_bayes(self):
        p=b.binary_filter([1,1,1],transition=0,error=.05)
        self.assertGreater(p[-1],.99)
        self.assertAlmostEqual(b.binary_filter([None])[0],.5)
    def test_affine(self):
        w=b.fit_affine(np.array([[1.],[2.],[3.],[4.]]))
        self.assertAlmostEqual(float(b.forecast_affine([4],w)[0]),5,delta=.05)
