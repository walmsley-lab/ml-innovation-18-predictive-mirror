import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import functional_state_interventions as m
from src import predictive_rgb_nca as base
from src import rgb_grounded_nca as grounded


class FunctionalTests(unittest.TestCase):
    def test_intact_matches_original_rollout(self):
        p=base.init(jax.random.PRNGKey(11),channels=4)
        x=jnp.zeros((2,3,8,8,3))
        a=m.forecast(p,x,horizon=2,ticks=1)
        b=base.rollout(p,x,horizon=2,ticks=1)
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=1e-6)

    def test_blend_matches_grounded_rollout(self):
        p=grounded.init(jax.random.PRNGKey(7),channels=4,output='blend')
        x=jnp.full((1,3,8,8,3),.1)
        expected=grounded.rollout(p,x,2,1,'blend')
        actual=m.forecast(p,x,2,1,'intact','blend')
        np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),atol=1e-6)

    def test_perturbation_content_and_guards(self):
        h=jnp.ones((1,8,8,4))
        image=jnp.zeros((1,8,8,3))
        z=m.perturb((h,image),"reset_hidden")
        np.testing.assert_array_equal(z[0],np.zeros((1,8,8,4)))
        with self.assertRaises(ValueError):
            m.perturb((h,image),"unknown")


if __name__=="__main__":
    unittest.main()
