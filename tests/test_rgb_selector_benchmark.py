"""Selector benchmark should use causally observed images and common baseline."""
import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from src import selectable_rgb_nca as nca, rgb_benchmark as benchmark
from experiments.rgb_selector_benchmark import run


class RGBSelectorBenchmarkTests(unittest.TestCase):
    def test_same_initialization_all_selectors_persist(self):
        params=nca.init(jax.random.PRNGKey(2),channels=4)
        x=np.random.default_rng(4).random((1,4,16,16,3),dtype=np.float32)
        for mode in nca.MODES:
            refs=nca.reference_map(16,16) if mode=="reference" else None
            pred=np.asarray(nca.rollout(params,jnp.asarray(x),horizon=3,
                                        ticks=1,mode=mode,references=refs))
            self.assertEqual(pred.shape,(1,3,16,16,3))
            np.testing.assert_allclose(pred,np.repeat(x[:,-1:],3,axis=1),
                                       atol=1e-6)

    def test_frozen_test_world_is_causal(self):
        x=benchmark.frames(benchmark.scenes(71000,1,context=4,horizon=3,size=16))
        controls=benchmark.controls(x[:,:4],3)
        self.assertIn("constant_velocity",controls)

    def test_step_zero_checkpoint_stays_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            report=run(folder,seeds=(3,),steps=1,context=3,horizon=2,
                       size=16,ticks=1,channels=4)
        for arm in report["runs"][0]["arms"].values():
            self.assertIn(arm["selected_step"],(0,1))
            self.assertLessEqual(arm["validation_mse"],
                                 arm["step_zero_validation_mse"])
