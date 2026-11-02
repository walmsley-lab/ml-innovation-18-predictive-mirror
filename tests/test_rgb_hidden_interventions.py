"""Hidden state interventions are causal, not alternate observations."""
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_hidden_interventions as probe
from experiments.rgb_hidden_interventions import run


class HiddenProbeTests(unittest.TestCase):
    def test_intact_matches_original_rollout(self):
        x=jnp.asarray(np.random.default_rng(3).random((1,5,16,16,3),dtype=np.float32))
        p=nca.init(jax.random.PRNGKey(3),4)
        original=np.asarray(nca.rollout(p,x,3,1))
        intact=np.asarray(probe.forecast(p,x,3,1,"intact",0))
        np.testing.assert_allclose(original,intact,atol=1e-6)

    def test_hidden_state_changes_but_rgb_anchor_is_identical(self):
        x=jnp.asarray(np.random.default_rng(2).random((1,5,16,16,3),dtype=np.float32))
        p=nca.init(jax.random.PRNGKey(4),4)
        state=probe.warm_state(p,x,1)
        for mode in ("reset","spatial_shuffle"):
            changed=probe.perturb(state,mode)
            np.testing.assert_array_equal(changed[1],state[1])
        self.assertGreater(float(jnp.abs(state[0]).sum()),0.)

    def test_untrained_outputs_persistence_and_no_peeking(self):
        x=np.random.default_rng(4).random((1,7,16,16,3),dtype=np.float32)
        p=nca.init(jax.random.PRNGKey(2),4)
        for mode in probe.MODES:
            f=np.asarray(probe.forecast(p,jnp.asarray(x[:,:4]),3,1,mode,1))
            np.testing.assert_allclose(f,np.repeat(x[:,3:4],3,axis=1),atol=1e-6)
            changed=x.copy();changed[:,4:]=0
            np.testing.assert_array_equal(f,np.asarray(
                probe.forecast(p,jnp.asarray(changed[:,:4]),3,1,mode,1)))

    def test_frozen_checkpoint_smoke(self):
        p=nca.init(jax.random.PRNGKey(3),4)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"weights.npz"
            np.savez_compressed(path,**jax.tree.map(np.asarray,p))
            r=run(path,Path(folder)/"out",episodes=1,context=4,horizon=3,size=16,
                  ticks=1,intervene_at=1)
            self.assertEqual(set(r["modes"]),set(probe.MODES))
            self.assertTrue((Path(folder)/"out"/"example.npz").exists())
            self.assertEqual(r["modes"]["reset"]["delta_from_intact_max_abs"],0.)


if __name__=="__main__":
    unittest.main()
