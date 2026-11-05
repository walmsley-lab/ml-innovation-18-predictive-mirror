import tempfile
import unittest
import numpy as np
import jax
from src import frozen_state_descriptions as f,predictive_rgb_nca as nca
from experiments.frozen_state_descriptions import run


class FrozenDescriptionTests(unittest.TestCase):
    def test_projection_frozen_train_only(self):
        p=nca.init(jax.random.PRNGKey(1),channels=6)
        frame=np.full((12,12,3),.2,np.float32)
        traces=np.stack([f.hidden_trace(p,frame,8),
                          f.hidden_trace(p,frame*.8,8)])
        self.assertEqual(traces.shape,(2,9,6))
        encoding,models=f.fit_descriptors(traces)
        self.assertEqual(f.project(traces,encoding).shape,(2,9,4))
        stats=f.evaluate(models,encoding,traces,3,3)
        self.assertIn("sindy",stats)
        self.assertTrue(np.isfinite(stats["sindy"]["latent_rmse"]))

    def test_requires_explicit_checkpoint(self):
        with self.assertRaises(ValueError):
            run(checkpoint=None,untrained_smoke=False)

    def test_smoke_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            report=run(folder+"/metrics.json",untrained_smoke=True,
                       channels=4,size=16,ticks=8,context=3,horizon=3)
            self.assertTrue(report["untrained_smoke"])
            self.assertIn("heldout_perturbation",report)


if __name__=="__main__":
    unittest.main()
