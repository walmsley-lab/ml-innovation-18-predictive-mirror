import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import convlstm_reference as model


class ConvLSTMTests(unittest.TestCase):
    def test_shapes_and_gradients(self):
        params=model.init(jax.random.PRNGKey(11),width=4)
        frames=jnp.zeros((2,4,8,8,3),jnp.float32)
        future=model.rollout(params,frames,3)
        self.assertEqual(tuple(future.shape),(2,3,8,8,3))
        self.assertTrue(np.isfinite(np.asarray(future)).all())
        examples=jnp.zeros((2,7,8,8,3),jnp.float32)
        value,grad=jax.value_and_grad(model.loss)(params,examples,4,3)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad)))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            model.rollout({},jnp.zeros((1,1,8,8,3)),1)


if __name__=="__main__":
    unittest.main()
