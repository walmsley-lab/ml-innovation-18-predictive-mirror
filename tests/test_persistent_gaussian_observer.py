import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from src import (persistent_gaussian_observer as model,
                 rgb_gaussian_nca as gaussian,world3d,
                 gaussian_observer_occlusion as occlusion)
from experiments.persistent_gaussian_observer import run


class PersistentGaussianObserverTests(unittest.TestCase):
    def test_causal_rollout_and_gradient(self):
        record=world3d.episode(11,7,16)
        params=gaussian.init(jax.random.PRNGKey(1),3,4)
        observed=jnp.asarray(record["rgb"][:4][None])
        actions=jnp.asarray(record["camera_actions"][None])
        pred=model.rollout(params,observed,actions,
                           jnp.asarray(record["K"]),3)
        self.assertEqual(pred.shape,(1,3,16,16,3))
        self.assertTrue(np.isfinite(np.asarray(pred)).all())
        target=jnp.asarray(record["rgb"][4:][None])
        value,grad=jax.value_and_grad(lambda p:jnp.mean(
            (model.rollout(p,observed,actions,
                           jnp.asarray(record["K"]),3)-target)**2))(params)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(all(np.isfinite(np.asarray(g)).all()
                            for g in jax.tree.leaves(grad)))

    def test_occlusion_changes_only_observations(self):
        record=world3d.episode(14,9,24)
        observed,coverage=occlusion.occlude(record,7,3)
        self.assertEqual(len(coverage),3)
        self.assertTrue(np.array_equal(observed[7:],record["rgb"][7:]))
        self.assertTrue(np.any(observed[:7]!=record["rgb"][:7]))

    def test_reset_and_short_baselines(self):
        record=world3d.episode(17,9,16)
        params=gaussian.init(jax.random.PRNGKey(4),3,4)
        rgb=jnp.asarray(record["rgb"][:5][None])
        actions=jnp.asarray(record["camera_actions"][None])
        K=jnp.asarray(record["K"])
        intact=model.rollout(params,rgb,actions,K,4)
        reset=model.rollout(params,rgb,actions,K,4,reset_at=4)
        short=gaussian.rollout(params,rgb,actions,K,4)
        self.assertEqual(intact.shape,reset.shape)
        self.assertEqual(short.shape,intact.shape)

    def test_smoke_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            report=run(folder,untrained_smoke=True,
                       seeds=(11,),context=5,horizon=2,size=16,
                       hidden_lengths=(2,))
            self.assertEqual(len(report["rows"]),1)
            self.assertIn("persistent",report["rows"][0]["scores"])


if __name__=="__main__":
    unittest.main()
