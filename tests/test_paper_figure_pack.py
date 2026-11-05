import unittest
import numpy as np
from experiments.paper_figure_pack import validate,metrics


class PaperFigureTests(unittest.TestCase):
    def setUp(self):
        self.true=np.zeros((2,3,8,8,3),np.float32)
        self.true[:,1,2,3,0]=1.
        self.data={"observed":np.zeros((2,8,8,3),np.float32),
                   "truth":self.true,"nca":self.true.copy(),
                   "persistence":np.zeros_like(self.true),
                   "velocity":np.zeros_like(self.true)}

    def test_source_arrays_and_mse(self):
        self.assertEqual(validate(self.data).shape,(2,3,8,8,3))
        result=metrics(self.data,(1,2,3))
        self.assertEqual(result["2"]["nca"],0.)
        self.assertGreater(result["2"]["persistence"],0.)

    def test_missing_or_incompatible_sources_rejected(self):
        del self.data["velocity"]
        with self.assertRaises(ValueError):
            validate(self.data)


if __name__=="__main__":
    unittest.main()
