import unittest
import jax
import numpy as np
from src import reference_growth_nca as growth


class GrowthTests(unittest.TestCase):
    def test_local_growth_and_damage_shapes(self):
        params=growth.init(jax.random.PRNGKey(11),channels=4,width=8)
        initial=growth.seed(size=8,channels=4)
        state=growth.rollout(params,initial,steps=2)
        self.assertEqual(tuple(state.shape),(1,8,8,4))
        self.assertEqual(tuple(growth.rgb(state).shape),(1,8,8,3))
        damaged=growth.damage(state,0.25)
        self.assertTrue(np.any(np.asarray(state)!=np.asarray(damaged)))

    def test_invalid_reference_settings(self):
        with self.assertRaises(ValueError):
            growth.seed(size=4)
        with self.assertRaises(ValueError):
            growth.rollout({},growth.seed(),0)


if __name__=="__main__":
    unittest.main()
