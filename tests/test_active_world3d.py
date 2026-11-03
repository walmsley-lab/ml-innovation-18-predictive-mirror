import tempfile
import unittest
import jax
import numpy as np
from src import active_world3d,action_rgb_nca
from experiments.active_world3d import run


class ActiveWorld3DTests(unittest.TestCase):
    def test_frozen_ensemble_policy_and_standardized_home(self):
        ensemble=[action_rgb_nca.init(jax.random.PRNGKey(i),4) for i in (1,2)]
        result=active_world3d.compare(ensemble,seeds=(5,),steps=3,size=16)
        self.assertEqual(set(result["scores"]),{"fixed","random","active"})
        self.assertEqual(result["paired"]["active_minus_fixed"]["pairs"],1)
        for v in result["scores"].values():
            self.assertTrue(np.isfinite(v["mean_home_mse"]))

    def test_untrained_requires_explicit_flag(self):
        with self.assertRaises(ValueError):
            run(checkpoints=(),untrained_smoke=False)
        with tempfile.TemporaryDirectory() as folder:
            report=run(out=folder+"/metrics.json",seeds=(2,),
                       steps=3,size=16,untrained_smoke=True)
            self.assertTrue(report["untrained_smoke"])


if __name__=="__main__":
    unittest.main()
