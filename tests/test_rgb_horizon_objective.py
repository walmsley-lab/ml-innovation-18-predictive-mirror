"""Late-lead objective emphasizes the required one-second prediction."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca
from experiments.rgb_horizon_objective import objective,horizon_weights


class HorizonObjectiveTests(unittest.TestCase):
    def test_weights_are_normalized_and_late_is_greater(self):
        first=np.asarray(horizon_weights(10,"uniform"))
        late=np.asarray(horizon_weights(10,"late"))
        self.assertAlmostEqual(first.mean(),1)
        self.assertAlmostEqual(late.mean(),1)
        self.assertGreater(late[-1],first[-1])
        self.assertLess(late[0],first[0])
        with self.assertRaises(ValueError):
            horizon_weights(1,"late")

    def test_objective_is_finite_for_tiny_scene(self):
        params=nca.init(jax.random.PRNGKey(4),channels=4)
        x=jnp.zeros((1,7,16,16,3))
        loss=objective(params,x,context=4,horizon=3,ticks=1,
                       weights=horizon_weights(3,"late"))
        self.assertTrue(np.isfinite(float(loss)))
