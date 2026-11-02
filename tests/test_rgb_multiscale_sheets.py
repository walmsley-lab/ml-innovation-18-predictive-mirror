"""Causal multi-timescale RGB NCA, step-zero selection, paired smoke."""
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_multiscale_sheets as nca
from experiments.rgb_multiscale_sheets import run


class MultiscaleTests(unittest.TestCase):
    def test_untrained_is_persistence_and_future_is_not_read(self):
        x=np.random.default_rng(3).random((1,7,16,16,3),dtype=np.float32)
        p=nca.init(jax.random.PRNGKey(2),4)
        for mode in nca.MODES:
            prediction=np.asarray(nca.rollout(p,jnp.asarray(x[:,:4]),3,1,mode))
            np.testing.assert_allclose(prediction,np.repeat(x[:,3:4],3,axis=1),atol=1e-6)
            changed=x.copy();changed[:,4:]=0
            np.testing.assert_array_equal(prediction,np.asarray(
                nca.rollout(p,jnp.asarray(changed[:,:4]),3,1,mode)))

    def test_same_weight_shape_and_finite_gradients(self):
        x=jnp.asarray(np.random.default_rng(2).random((1,7,16,16,3),dtype=np.float32))
        p=nca.init(jax.random.PRNGKey(3),4)
        for mode in nca.MODES:
            error,grad=jax.value_and_grad(nca.loss)(p,x,4,3,1,mode)
            self.assertTrue(np.isfinite(float(error)))
            self.assertTrue(all(bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(grad)))

    def test_matched_training_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run(folder,seeds=(3,),steps=1,context=4,horizon=2,
                       size=16,ticks=1,channels=4)
            self.assertEqual(set(result["runs"][0]["arms"]),set(nca.MODES))
            self.assertTrue((Path(folder)/"seed-3-multiscale.npz").exists())
            self.assertIn("step_zero_validation_mse",result["runs"][0]["arms"]["single"])


if __name__=="__main__":
    unittest.main()
