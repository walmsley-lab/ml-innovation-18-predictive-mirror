"""Phase entrainment: nontrivial sensory response, no peeking and smoke."""
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_phase_entrainment as phase
from src import predictive_rgb_nca as base
from experiments.rgb_phase_entrainment import run


class EntrainmentTests(unittest.TestCase):
    def test_heterogeneous_phase_reacts_to_observed_surprise(self):
        state=jnp.zeros((1,16,16,1))
        image=jnp.zeros((1,16,16,3))
        driven=phase.evolve(state,image,jnp.ones_like(image),"entrained")
        idle=phase.evolve(state,image,jnp.zeros_like(image),"entrained")
        self.assertGreater(float(jnp.abs(driven-idle).mean()),.1)
        omega=phase.natural_frequency(16,16)
        self.assertGreater(float(jnp.max(omega)-jnp.min(omega)),.01)

    def test_zero_readout_persists_and_future_is_inaccessible(self):
        frames=np.random.default_rng(4).random((1,7,16,16,3),dtype=np.float32)
        weights=base.init(jax.random.PRNGKey(4),4)
        for mode in phase.MODES:
            f=np.asarray(phase.rollout(weights,jnp.asarray(frames[:,:4]),3,1,mode))
            np.testing.assert_allclose(f,np.repeat(frames[:,3:4],3,axis=1),atol=1e-6)
            altered=frames.copy();altered[:,4:]=0
            np.testing.assert_array_equal(f,np.asarray(
                phase.rollout(weights,jnp.asarray(altered[:,:4]),3,1,mode)))

    def test_training_gradients_and_smoke(self):
        examples=jnp.asarray(np.random.default_rng(3).random((1,7,16,16,3),dtype=np.float32))
        weights=base.init(jax.random.PRNGKey(2),4)
        grads=jax.grad(phase.loss)(weights,examples,4,3,1,"entrained")
        self.assertTrue(all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grads)))
        with tempfile.TemporaryDirectory() as folder:
            result=run(folder,seeds=(3,),steps=1,context=4,horizon=2,size=16,ticks=1,channels=4)
            self.assertEqual(set(result["results"][0]["arms"]),set(phase.MODES))
            self.assertTrue((Path(folder)/"seed-3-entrained.npz").exists())
            self.assertIsNone(result["results"][0]["arms"]["plain"]["phase_global_order"])
            for arm in result["results"][0]["arms"].values():
                self.assertLessEqual(arm["selected_validation_mse"],arm["step_zero_validation_mse"])


if __name__=="__main__":
    unittest.main()
