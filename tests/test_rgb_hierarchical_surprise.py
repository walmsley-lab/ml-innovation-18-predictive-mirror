import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_hierarchical_surprise as model
from experiments.rgb_hierarchical_surprise import run


class HierarchicalSurpriseTests(unittest.TestCase):
    def test_shape_and_causality(self):
        params=model.init(jax.random.PRNGKey(1),channels=4)
        context=jnp.ones((1,3,16,16,3),jnp.float32)*.2
        for mode in model.MODES:
            pred,latent=model.rollout(params,context,horizon=2,mode=mode,
                                     return_latent=True)
            self.assertEqual(pred.shape,(1,2,16,16,3))
            self.assertEqual(latent.shape,(1,2,16,16,4))
        with self.assertRaises(ValueError):
            model.rollout(params,context[:,:,:15],2)

    def test_loss_is_finite(self):
        params=model.init(jax.random.PRNGKey(2),channels=4)
        sample=jnp.ones((1,5,16,16,3),jnp.float32)*.2
        for mode in model.MODES:
            self.assertTrue(np.isfinite(float(model.loss(params,sample,3,2,mode))))

    def test_one_step_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            result=run(out=temp,seeds=(3,),steps=1,context=3,
                       horizon=2,size=16,batch_size=2,channels=4)
            self.assertEqual(len(result["results"]),1)
            self.assertTrue((Path(temp)/"metrics.json").exists())


if __name__=="__main__":
    unittest.main()
