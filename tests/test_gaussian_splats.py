import tempfile
import unittest
import numpy as np
from src import world3d,gaussian_splats as gs
from experiments.gaussian_oracle_baselines import run


class GaussianSplatTests(unittest.TestCase):
    def test_render_is_finite_and_pose_changes_view(self):
        w=world3d.episode(12,7,24)
        state=gs.state_from_record(w,3)
        a=gs.render(state["center"],state["scale"],state["color"],
                    state["opacity"],w["K"],w["T_world_camera"][2],24)
        translated=w["T_world_camera"][2].copy()
        translated[0,3]+=.4
        b=gs.render(state["center"],state["scale"],state["color"],
                    state["opacity"],w["K"],translated,24)
        self.assertEqual(a.shape,(24,24,3))
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(float(np.mean(abs(a-b))),0)

    def test_oracle_and_state_rules(self):
        w=world3d.episode(18,8,24)
        records=gs.evaluate(w,3,3)
        self.assertEqual(set(records),{"static","velocity","oracle_future_geometry"})
        self.assertEqual(records["oracle_future_geometry"]["centroid_3d_mae_m"],0)
        self.assertEqual(records["oracle_future_geometry"]["mse_vs_gaussian_oracle"],0)
        with self.assertRaises(ValueError):
            gs.state_from_record(w,1)

    def test_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run(folder+"/metrics.json",seeds=(1,),context=3,horizon=2,size=16)
            self.assertIn("privileged",result["protocol"])


if __name__=="__main__":
    unittest.main()
