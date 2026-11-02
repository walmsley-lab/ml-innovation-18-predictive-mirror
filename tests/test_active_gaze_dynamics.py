"""Active gaze: policy has no truth input, action alignment and controls."""
import unittest
import numpy as np
from src import active_gaze_dynamics as gaze


class ActiveGazeTests(unittest.TestCase):
    def test_bounce_and_distribution_normalized(self):
        self.assertEqual(gaze.bounce(0,-1),(1,1))
        self.assertEqual(gaze.bounce(gaze.WORLD-1,1),(gaze.WORLD-2,-1))
        prior=np.ones(len(gaze.STATES))/len(gaze.STATES)
        future=gaze.step_belief(prior)
        self.assertAlmostEqual(float(future.sum()),1.)
        self.assertAlmostEqual(float(gaze.future_position(prior,10).sum()),1.)

    def test_policy_does_not_take_hidden_target(self):
        belief=np.ones(len(gaze.STATES))/len(gaze.STATES)
        action=gaze.choose_view(belief,gaze.START,5)
        self.assertIn(action,gaze.ACTIONS)
        before=gaze.expected_risk(belief,gaze.START,5)
        self.assertTrue(np.isfinite(before))

    def test_reproducible_sensor_and_matched_observations(self):
        a=gaze.episode(120,"active",context=5,horizon=3)
        b=gaze.episode(120,"active",context=5,horizon=3)
        self.assertEqual(a,b)
        self.assertEqual(len(a["views"]),5)
        self.assertEqual(set(a["predictions"]),{"action_aware","action_blind"})
        self.assertNotIn("target_future_x",a["views"][0])

    def test_comparison_and_different_view_policies(self):
        report=gaze.compare(seeds=(3,),episodes=3,context=5,horizon=3)
        self.assertEqual(set(report["policies"]),{"fixed","random","active"})
        for policy,detail in report["policies"].items():
            self.assertEqual(detail["runs"],3)
            self.assertTrue(np.isfinite(detail["mean_position_error"]["action_aware"]))
            self.assertIn("local_observation",detail["sample_trace"]["views"][0])


if __name__=="__main__":
    unittest.main()
