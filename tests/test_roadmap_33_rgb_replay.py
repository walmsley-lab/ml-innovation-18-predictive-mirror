import unittest
import numpy as np
import jax
import jax.numpy as jnp
from experiments import roadmap_33_rgb_replay as m
from src import predictive_rgb_nca as nca


class RGBReplayTests(unittest.TestCase):
    def test_no_future_frame_in_committed_forecast(self):
        example=m.example(31000,"visible_motion",context=3,size=16)
        params=nca.init(jax.random.PRNGKey(5),2)
        first=np.asarray(nca.rollout(params,jnp.asarray(example[:3][None]),1,1))
        changed=example.copy()
        changed[3]=1-changed[3]
        second=np.asarray(nca.rollout(params,jnp.asarray(changed[:3][None]),1,1))
        np.testing.assert_allclose(first,second)

    def test_equal_updates_and_replay_history(self):
        for mode in ("replay","recent_only"):
            result=m.learn(3,mode,capacity=3,stage_lengths=(2,2,1),
                           context=3,size=16,channels=2,ticks=1)
            self.assertEqual(result["gradient_updates"],10)
            self.assertEqual(result["total_examples"],5)
            self.assertEqual(len(result["stage_metrics"]),3)
            self.assertIn("occluded_camera",
                          result["stage_metrics"][1]["heldout_test"])

    def test_invalid_domain(self):
        with self.assertRaises(ValueError):
            m.example(1,"oracle",3,16)


if __name__=="__main__":
    unittest.main()
