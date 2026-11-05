import unittest
import numpy as np
from experiments import roadmap_63_retinotopy as m


class RetinotopyTests(unittest.TestCase):
    def test_constant_signal_is_preserved(self):
        p=m.toy_positions(8)
        x=np.ones((3,8))
        np.testing.assert_allclose(m.rasterize(x,p,size=12),1,atol=1e-8)

    def test_external_cannot_infer_registration(self):
        with self.assertRaises(ValueError):
            m.validate(np.ones((8,2)),9)
        with self.assertRaises(ValueError):
            m.run(matrix="foo",metadata=None,out=None)

    def test_permutation_has_nonzero_effect(self):
        r=m.run(out=None,seeds=(11,))
        self.assertEqual(r["source_type"],"synthetic")
        self.assertGreater(r["rows"][0]["layout_effect_mse"],0)


if __name__=="__main__":
    unittest.main()
