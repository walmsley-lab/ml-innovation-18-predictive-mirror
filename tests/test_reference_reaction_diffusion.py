import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import reference_reaction_diffusion as rd


class ReactionDiffusionTests(unittest.TestCase):
    def test_reproducible_stable_world(self):
        a=rd.sequence(11,16,frames=5,burnin=1,substeps=2)
        b=rd.sequence(11,16,frames=5,burnin=1,substeps=2)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(5,16,16,2))
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(float(np.mean(np.abs(a[0]-a[-1]))),0)

    def test_constant_field_has_zero_laplacian(self):
        self.assertAlmostEqual(float(np.max(np.abs(rd.laplace(np.ones((16,16)))))),0.)

    def test_shared_local_forecast_shape(self):
        p=rd.init(jax.random.PRNGKey(11),width=4)
        x=jnp.asarray(rd.start(size=16)[None])
        out=rd.rollout(p,x,3)
        self.assertEqual(tuple(out.shape),(1,3,16,16,2))
        self.assertTrue(np.isfinite(np.asarray(out)).all())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            rd.sequence(frames=1)
        with self.assertRaises(ValueError):
            rd.rollout({},jnp.zeros((1,16,16,2)),0)


if __name__=="__main__":
    unittest.main()
