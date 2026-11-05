import tempfile
import unittest
import numpy as np
from src import layered_advection_world as world
from experiments.layered_advection_benchmark import run


class LayeredWorldTests(unittest.TestCase):
    def test_all_domains_and_masks(self):
        for mode in world.MODES:
            rec=world.episode(2,mode=mode)
            counts=world.validate(rec)
            self.assertEqual(counts["frames"],12)
            self.assertGreaterEqual(counts["target_disoccluded_pixels"],0)
            self.assertEqual(rec["instance_ids"].dtype,np.int16)
            self.assertEqual(world.model_inputs(rec,4).shape,(4,32,32,3))

    def test_subpixel_motion_and_stationary_flicker(self):
        sub=world.episode(4,mode="subpixel")
        self.assertTrue(np.all(np.abs(sub["velocities"])<1))
        still=world.episode(4,mode="appearance_only")
        self.assertEqual(float(np.max(abs(still["velocities"]))),0.)
        self.assertGreater(np.mean(abs(still["frames"][0]-
                                       still["frames"][1])),1e-3)

    def test_depth_hiding_and_reproducibility(self):
        a=world.episode(11,steps=15)
        b=world.episode(11,steps=15)
        np.testing.assert_array_equal(a["frames"],b["frames"])
        np.testing.assert_array_equal(a["instance_ids"],b["instance_ids"])
        self.assertGreater(world.validate(a)["source_occluded_pixels"],0)

    def test_runner_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run(folder,seeds=(1,),modes=("subpixel",),steps=5)
            self.assertEqual(len(result["rows"]),1)


if __name__=="__main__":
    unittest.main()
