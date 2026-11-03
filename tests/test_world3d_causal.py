import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import action_rgb_nca,world3d_causal
from experiments.world3d_causal import run


class Causal3DTests(unittest.TestCase):
    def test_intact_matches_original_and_lesion_changes_hidden(self):
        params=action_rgb_nca.init(jax.random.PRNGKey(1),4)
        context=jnp.ones((2,3,16,16,3),jnp.float32)*.25
        actions=jnp.ones((2,4,2),jnp.float32)*.12
        original=action_rgb_nca.rollout(params,context,actions,2)
        intact=world3d_causal.rollout(params,context,actions,2)
        np.testing.assert_allclose(np.asarray(original),np.asarray(intact),atol=1e-6)
        lesion=world3d_causal.rollout(params,context,actions,2,"zero_hidden")
        self.assertEqual(lesion.shape,original.shape)

    def test_untrained_smoke_explicit(self):
        with self.assertRaises(ValueError):
            run(episodes=2)
        with tempfile.TemporaryDirectory() as folder:
            report=run(out=folder+"/metrics.json",seed=41,episodes=2,
                       context=3,horizon=2,size=16,untrained_smoke=True)
            self.assertTrue(report["untrained_smoke"])
            self.assertEqual(len(report["domains"]),4)


if __name__=="__main__":
    unittest.main()
