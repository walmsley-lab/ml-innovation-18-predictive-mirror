import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import recurrent_transport_nca as model
from experiments.recurrent_transport_nca import run


class RecurrentTransportTests(unittest.TestCase):
    def test_all_arms_are_open_loop_and_bounded(self):
        p=model.init(jax.random.PRNGKey(1),4)
        observed=jnp.ones((2,3,16,16,3))*0.3
        for mode in model.MODES:
            prediction=model.rollout(p,observed,2,mode)
            self.assertEqual(prediction.shape,(2,2,16,16,3))
            self.assertGreaterEqual(float(prediction.min()),0)
            self.assertLessEqual(float(prediction.max()),1)
        with self.assertRaises(ValueError):
            model.rollout(p,observed,2,"oracle")

    def test_one_step_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run(folder,seeds=(1,),steps=1,context=3,horizon=2,
                       size=16,width=4,batch_size=2)
            self.assertEqual(set(result["results"][0]["modes"]),set(model.MODES))


if __name__ == "__main__":
    unittest.main()
