import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import field_multiscale as m
from experiments.field_multiscale_comparison import run


class MultiScaleFieldTests(unittest.TestCase):
    def test_averaging_and_dimension_guard(self):
        x=jnp.arange(4*4,dtype=jnp.float32).reshape(1,4,4,1)
        self.assertEqual(m.average2(x).shape,(1,2,2,1))
        self.assertEqual(m.up2(m.average2(x)).shape,x.shape)
        with self.assertRaises(ValueError):
            m.average2(jnp.ones((1,5,4,1)))

    def test_all_modes_and_direction_ablations(self):
        params=m.init(jax.random.PRNGKey(1),4)
        x=jnp.zeros((2,16,16,2))
        scores=m.intervention(params,x,2,ticks=1)
        self.assertEqual(set(scores),set(m.MODES))
        self.assertTrue(all(np.isfinite(np.asarray(y)).all() for y in scores.values()))
        for y in scores.values():
            self.assertEqual(y.shape,(2,2,16,16,2))

    def test_short_training(self):
        with tempfile.TemporaryDirectory() as folder:
            r=run(folder,kind="diffusion",seeds=(1,),steps=1,horizon=2,
                  size=16,channels=4,ticks=1,batch_size=2)
            self.assertEqual(len(r["results"]),5)


if __name__=="__main__":
    unittest.main()
