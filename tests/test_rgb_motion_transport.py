import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from experiments.rgb_motion_transport import run
from src import rgb_motion_transport as model


class RGBMotionTransportTests(unittest.TestCase):
    def test_nonwrapping_candidates_and_probabilities(self):
        image=jnp.zeros((1,16,16,3),jnp.float32).at[0,0,0,:].set(1)
        choices=np.asarray(model.candidates(image))
        self.assertEqual(choices.shape,(1,16,16,9,3))
        self.assertAlmostEqual(float(choices[0,-1,-1].max()),0)
        param=model.init(jax.random.PRNGKey(2),width=4)
        _,weights=model.one_step(param,image,image,"combined",True)
        np.testing.assert_allclose(np.asarray(weights.sum(axis=-1)),1,atol=1e-6)

    def test_rollout_uses_context_alone(self):
        p=model.init(jax.random.PRNGKey(1),width=4)
        context=jnp.ones((2,3,16,16,3),jnp.float32)*.2
        self.assertEqual(model.rollout(p,context,horizon=2).shape,(2,2,16,16,3))

    def test_single_step_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            report=run(out=temp,seeds=(3,),steps=1,context=3,horizon=2,
                       size=16,batch_size=2,width=4)
            self.assertEqual(len(report["results"]),1)
            self.assertTrue((Path(temp)/"metrics.json").exists())


if __name__=="__main__":
    unittest.main()
