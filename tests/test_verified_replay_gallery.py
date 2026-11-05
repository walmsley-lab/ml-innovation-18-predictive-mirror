import unittest
import numpy as np
from experiments.verified_replay_gallery import panel_arrays


class ReplayGalleryTests(unittest.TestCase):
    def test_error_panel_matches_absolute_difference(self):
        x=np.zeros((8,8,3),np.float32)
        y=np.ones_like(x)*.2
        z=np.ones_like(x)*.5
        p=panel_arrays(x,y,z)
        np.testing.assert_allclose(p[3],np.ones_like(x)*.3,atol=1e-7)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            panel_arrays(np.zeros((4,4,3)),np.zeros((8,8,3)),np.zeros((8,8,3)))


if __name__=="__main__":
    unittest.main()
