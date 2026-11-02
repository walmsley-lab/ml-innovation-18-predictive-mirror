"""Soft-reference selection tests: nonwrapping, causality, gradient and smoke."""
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_soft_references as model
from experiments.rgb_soft_references import run


class SoftReferenceTests(unittest.TestCase):
    def test_reference_candidates_never_wrap(self):
        x=jnp.zeros((1,16,16,7))
        x=x.at[0,0,0,0].set(1.)
        candidates=np.asarray(model.candidates(x))
        self.assertEqual(candidates.shape,(1,16,16,len(model.OFFSETS),7))
        self.assertEqual(float(candidates[0,-1,-1].sum()),0.)

    def test_adaptive_routing_and_shuffled_control(self):
        p=model.init(jax.random.PRNGKey(3),4)
        x=jnp.asarray(np.random.default_rng(2).random((1,16,16,10),dtype=np.float32))
        learned,w=model.choose(p,x,"learned")
        shuffled,other=model.choose(p,x,"shuffled")
        self.assertEqual(learned.shape,x.shape)
        np.testing.assert_allclose(np.asarray(w.sum(axis=-1)),1.,atol=1e-5)
        self.assertGreater(float(jnp.max(jnp.abs(learned-shuffled))),0.)

    def test_untrained_equals_persistence_and_no_peeking(self):
        x=np.random.default_rng(5).random((1,7,16,16,3),dtype=np.float32)
        p=model.init(jax.random.PRNGKey(3),4)
        for mode in model.MODES:
            prediction=np.asarray(model.rollout(p,jnp.asarray(x[:,:4]),3,1,mode))
            np.testing.assert_allclose(prediction,
                np.repeat(x[:,3:4],3,axis=1),atol=1e-6)
            changed=x.copy();changed[:,4:]=1
            np.testing.assert_array_equal(prediction,np.asarray(
                model.rollout(p,jnp.asarray(changed[:,:4]),3,1,mode)))

    def test_gradients_and_smoke(self):
        x=jnp.asarray(np.random.default_rng(6).random((1,7,16,16,3),dtype=np.float32))
        p=model.init(jax.random.PRNGKey(4),4)
        grad=jax.grad(model.loss)(p,x,4,3,1,"learned")
        self.assertTrue(all(bool(jnp.isfinite(y).all()) for y in jax.tree.leaves(grad)))
        with tempfile.TemporaryDirectory() as d:
            result=run(d,seeds=(3,),steps=1,context=4,horizon=2,size=16,ticks=1,channels=4)
            self.assertEqual(set(result["runs"][0]["arms"]),set(model.MODES))
            self.assertTrue((Path(d)/"seed-3-learned.npz").exists())


if __name__=="__main__":
    unittest.main()
