import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_gaussian_nca as model,world3d
from experiments.rgb_gaussian_nca import run


class RGBGaussianNCATests(unittest.TestCase):
    def setUp(self):
        worlds=[world3d.episode(i,5,16) for i in (1,2)]
        self.rgb=jnp.asarray(np.stack([w["rgb"] for w in worlds]))
        self.actions=jnp.asarray(np.stack([w["camera_actions"] for w in worlds]))
        self.K=jnp.asarray(worlds[0]["K"])
        self.p=model.init(jax.random.PRNGKey(2),3,4)

    def test_shapes_and_finite_grad(self):
        outputs=model.rollout(self.p,self.rgb[:,:3],self.actions,self.K,2)
        self.assertEqual(outputs.shape,(2,2,16,16,3))
        self.assertTrue(np.isfinite(np.asarray(outputs)).all())
        error,grads=jax.value_and_grad(model.loss)(
            self.p,self.rgb,self.actions,self.K,3,2)
        self.assertTrue(np.isfinite(float(error)))
        self.assertTrue(all(np.isfinite(np.asarray(g)).all()
                            for g in jax.tree.leaves(grads)))

    def test_known_future_camera_action_changes_render(self):
        actions=self.actions.at[:,2:,0].set(.35)
        x=model.rollout(self.p,self.rgb[:,:3],self.actions,self.K,2)
        y=model.rollout(self.p,self.rgb[:,:3],actions,self.K,2)
        self.assertGreater(float(np.mean(np.abs(np.asarray(x-y)))),1e-7)

    def test_short_cpu_training(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run(folder,seeds=(1,),steps=1,context=3,horizon=2,
                       size=16,slots=3,width=4,batch_size=2)
            self.assertEqual(len(result["runs"]),1)
            self.assertIn("cellular",result["runs"][0]["modes"])


if __name__=="__main__":
    unittest.main()
