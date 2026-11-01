"""Phase state must be nontrivial, causal and parameter-budget matched."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_phase_nca as phase


class RGBPhaseNCATests(unittest.TestCase):
    def test_coupling_changes_phase(self):
        initial=phase.initial_phase(1,16,16)
        frame=jnp.zeros((1,16,16,3),jnp.float32)
        uncoupled=phase.phase_step(initial,frame,0.)
        coupled=phase.phase_step(initial,frame,.25)
        self.assertGreater(float(jnp.abs(coupled-uncoupled).max()),1e-5)

    def test_initial_readout_is_persistence_in_all_modes(self):
        params=nca.init(jax.random.PRNGKey(4),channels=4)
        x=np.random.default_rng(5).random((1,4,16,16,3),dtype=np.float32)
        for mode in phase.MODES:
            future=np.asarray(phase.rollout(params,jnp.asarray(x),3,1,mode))
            np.testing.assert_allclose(future,np.repeat(x[:,-1:],3,axis=1),
                                       atol=1e-6)
