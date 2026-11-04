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
            measurement=report["runs"][0]["measurement"]
            self.assertIn("clip_rail_saturation",measurement)
            self.assertIn("measurement_verdict",measurement)
            # Counterfactual action plans (positive control) and the world's own
            # action dependence (learnable ceiling) are both reported.
            self.assertIn("zeroed_actions",measurement["positive_control_response"])
            self.assertEqual(set(measurement["world_action_dependence"]),
                             {str(h) for h in (1,2)})

    def test_counterfactual_actions_move_a_live_model(self):
        # A model whose readout actually reads the action-fed state changes its
        # forecast when the planned actions change: the action wire is live.
        p=model.init(jax.random.PRNGKey(4),6)
        p=dict(p,readout=p["readout"]*4.0)
        rgb=jnp.asarray(np.random.default_rng(1).random(
            (3,3,16,16,3),dtype=np.float32))
        actions=jnp.asarray(np.random.default_rng(2).uniform(
            -.3,.3,(3,4,2)).astype(np.float32))
        planned=np.asarray(model.rollout(p,rgb,actions,2,True))
        zeroed=np.asarray(model.rollout(p,rgb,jnp.zeros_like(actions),2,True))
        self.assertGreater(float(np.abs(planned-zeroed).mean()),1e-5)


if __name__=="__main__":
    unittest.main()
