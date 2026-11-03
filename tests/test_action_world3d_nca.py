import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import action_rgb_nca as model
from experiments.action_world3d_nca import worlds,run


class ActionWorld3DTests(unittest.TestCase):
    def test_causal_shapes(self):
        p=model.init(jax.random.PRNGKey(1),4)
        rgb=jnp.ones((2,3,16,16,3))*.2
        actions=jnp.zeros((2,4,2))
        a=model.rollout(p,rgb,actions,2,True)
        b=model.rollout(p,rgb,actions,2,False)
        self.assertEqual(a.shape,(2,2,16,16,3))
        np.testing.assert_allclose(np.asarray(a),np.asarray(b))
        with self.assertRaises(ValueError):
            model.rollout(p,rgb,actions[:,:2],2)

    def test_dataset_contract(self):
        videos,actions,records=worlds(2,2,5,16,"train")
        self.assertEqual(videos.shape,(2,5,16,16,3))
        self.assertEqual(actions.shape,(2,4,2))
        self.assertIn("depth",records[0])

    def test_one_step_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            report=run(folder,seeds=(2,),steps=1,context=3,horizon=2,
                       size=16,width=4,batch_size=2)
            self.assertEqual(set(report["runs"][0]["arms"]),
                             {"action_aware","action_blind"})


if __name__=="__main__":
    unittest.main()
