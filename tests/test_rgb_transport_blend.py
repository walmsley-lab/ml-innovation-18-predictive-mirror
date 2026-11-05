import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_transport_blend as model


class TransportBlendTests(unittest.TestCase):
    def test_bounded_and_component_shapes(self):
        p=model.init(jax.random.PRNGKey(11),width=4)
        before=jnp.zeros((1,8,8,3),jnp.float32)
        current=jnp.full_like(before,.4)
        future,parts=model.one_step(p,before,current,return_parts=True)
        self.assertEqual(tuple(parts["synthesis_gate"].shape),(1,8,8,1))
        self.assertTrue(np.isfinite(np.asarray(future)).all())
        self.assertGreaterEqual(float(future.min()),0.)
        self.assertLessEqual(float(future.max()),1.)
        for mode in model.MODES:
            out=model.rollout(p,jnp.stack([before,current],axis=1),3,mode)
            self.assertEqual(tuple(out.shape),(1,3,8,8,3))

    def test_invalid_modes(self):
        with self.assertRaises(ValueError):
            model.one_step({},None,None,"wrong")


if __name__=="__main__":
    unittest.main()
