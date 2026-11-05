import unittest
import numpy as np
from experiments import development_27_egomotion as exp


class Development27Tests(unittest.TestCase):
    def test_observer_action_is_realized(self):
        frames, actions, positions, offsets = exp.world(13)
        np.testing.assert_array_equal(actions,np.diff(offsets))
        self.assertEqual(exp.world_velocity(frames,actions).shape,(len(actions),))
        np.testing.assert_allclose(exp.world_velocity(frames,actions),
                                   np.diff(positions[:,0]),atol=.1)

    def test_registration_and_causal_prefix(self):
        frames, actions, _, _ = exp.world(11)
        self.assertLessEqual(abs(exp.register_background(frames[1],frames[2])),2)
        np.testing.assert_allclose(exp.world_velocity(frames[:6],actions[:5]),
                                   exp.world_velocity(frames,actions)[:5])

    def test_controls(self):
        r=exp.run(out=None)
        self.assertIn("image_registration",r["scores"])
        self.assertIn("shuffled_action",r["scores"])
        with self.assertRaises(ValueError):
            exp.world_velocity(exp.world()[0],mode="action")


if __name__=="__main__":
    unittest.main()
