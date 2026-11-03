import tempfile
import unittest
import numpy as np
from src import world3d
from experiments.world3d_generate import run


class World3DTests(unittest.TestCase):
    def test_calibrated_shapes_and_no_input_truth(self):
        rec=world3d.episode(seed=1,steps=5,size=24)
        self.assertEqual(rec["rgb"].shape,(5,24,24,3))
        self.assertEqual(rec["depth"].shape,(5,24,24))
        self.assertEqual(rec["camera_actions"].shape,(4,2))
        self.assertEqual(rec["T_world_camera"].shape,(5,4,4))
        self.assertEqual(set(world3d.model_inputs(rec)),{"rgb"})
        self.assertEqual(set(world3d.model_inputs(rec,True)),
                         {"rgb","camera_actions"})
        self.assertTrue(np.isfinite(rec["depth"][rec["instance_id"]>=0]).all())

    def test_projection_parallax_and_depth(self):
        K=world3d.intrinsics(48)
        points=np.array([[.5,0,3],[.5,0,6]],np.float32)
        uv0,z0=world3d.project(points,np.zeros(3),0.,K)
        uv1,_=world3d.project(points,np.array([.2,0,0]),0.,K)
        self.assertGreater(abs(uv1[0,0]-uv0[0,0]),
                           abs(uv1[1,0]-uv0[1,0]))
        self.assertLess(z0[0],z0[1])

    def test_deterministic_and_manifest(self):
        a=world3d.episode(3,steps=4,size=16)
        b=world3d.episode(3,steps=4,size=16)
        np.testing.assert_array_equal(a["rgb"],b["rgb"])
        with tempfile.TemporaryDirectory() as folder:
            m=run(folder,episodes=1,steps=4,size=16)
            self.assertEqual(m["backend"],"ray-sphere-analytic-v1")


if __name__=="__main__":
    unittest.main()
