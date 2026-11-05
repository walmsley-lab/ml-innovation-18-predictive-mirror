import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import gaussian_cellular_dynamics as model
from experiments.gaussian_oracle_dynamics import worlds,run


class GaussianCellularTests(unittest.TestCase):
    def test_gradient_and_no_future_input(self):
        p=model.init(jax.random.PRNGKey(1),6)
        center=jnp.ones((2,3,3),jnp.float32)
        vel=jnp.ones_like(center)*.02
        pred=model.rollout(p,center,vel,3)
        self.assertEqual(pred.shape,(2,3,3,3))
        truth=jnp.zeros_like(pred)
        value,grad=jax.value_and_grad(model.loss)(p,center,vel,truth)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(all(np.isfinite(np.asarray(g)).all()
                            for g in jax.tree.leaves(grad)))

    def test_oracle_input_only_past(self):
        a,b,f,records=worlds(1,2,3,2,16,"train")
        self.assertEqual(a.shape,(2,3,3))
        self.assertEqual(f.shape,(2,2,3,3))
        self.assertIn("rgb",records[0])

    def test_one_step_training(self):
        with tempfile.TemporaryDirectory() as folder:
            report=run(folder,seeds=(1,),steps=1,context=3,horizon=2,
                       size=16,channels=4,batch_size=2)
            self.assertEqual(report["results"][0]["seed"],1)


if __name__=="__main__":
    unittest.main()
