import tempfile
import unittest
import jax
import numpy as np
from src import gaussian_active_view as gaze,rgb_gaussian_nca as model
from experiments.gaussian_active_view import run


class GaussianActiveTests(unittest.TestCase):
    def test_action_budget_and_yoke_requirement(self):
        ensemble=[model.init(jax.random.PRNGKey(x),2,3) for x in (1,2)]
        with self.assertRaises(ValueError):
            gaze.episode(3,ensemble,"yoked",steps=3,size=16)

    def test_untrained_evaluation_is_explicit(self):
        with self.assertRaises(ValueError):
            run(checkpoints=(),untrained_smoke=False)
        with tempfile.TemporaryDirectory() as folder:
            report=run(out=folder+"/metrics.json",seeds=(2,),
                       domains=("ood_depth",),steps=3,size=16,
                       untrained_smoke=True)
            self.assertTrue(report["untrained_smoke"])
            rows=report["domains"]["ood_depth"]["episodes"][0]["policies"]
            self.assertEqual(set(rows),set(gaze.POLICIES))
            self.assertEqual(set(x["observations"] for x in rows.values()),{3})
            self.assertGreaterEqual(rows["active"]["selection_forecast_calls"],0)


if __name__=="__main__":
    unittest.main()
