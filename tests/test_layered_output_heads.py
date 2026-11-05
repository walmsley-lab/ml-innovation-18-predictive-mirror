import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from src import layered_output_heads as h,layered_advection_world as w
from experiments.layered_output_comparison import oracle_probe,run


class LayeredOutputTests(unittest.TestCase):
    def test_gather_direction_nonwrapping_and_grad(self):
        a=jnp.zeros((1,12,12,3)).at[0,3,4,0].set(1)
        delta=jnp.broadcast_to(jnp.asarray([1.,2.]),(1,12,12,2))
        out,valid=h.gather(a,delta)
        self.assertAlmostEqual(float(out[0,4,6,0]),1)
        self.assertFalse(bool(valid[0,0,0]))
        p=h.init(jax.random.PRNGKey(1),4)
        sample=jnp.ones((1,3,12,12,3))*.13
        for mode in h.MODES:
            forecast=h.rollout(p,sample,2,mode)
            self.assertEqual(forecast.shape,(1,2,12,12,3))
            self.assertTrue(np.isfinite(np.asarray(forecast)).all())

    def test_oracle_factorial_never_training_input(self):
        r=w.episode(2,steps=7)
        p=h.init(jax.random.PRNGKey(1),4)
        diag=oracle_probe(p,r,4,"transport_synthesis")
        self.assertEqual(len(diag),4)
        self.assertIn("flow_1_visible_1",diag)

    def test_one_update_all_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            report=run(directory,seeds=(1,),steps=1,context=4,
                       horizon=2,size=20,width=4,batch_size=2)
            self.assertEqual(len(report["results"]),3)


if __name__=="__main__":
    unittest.main()
