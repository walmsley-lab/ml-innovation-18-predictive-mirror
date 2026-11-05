import unittest
import numpy as np
from experiments import development_28_action as exp


class Development28Tests(unittest.TestCase):
    def test_no_future_rgb_in_feature(self):
        record=exp.episode(11)
        original=exp.feature(record,3)
        altered=dict(record)
        altered["frames"]=record["frames"].copy()
        altered["frames"][4:]=0.
        np.testing.assert_array_equal(original,exp.feature(altered,3))

    def test_action_alignment_and_equal_capacity(self):
        record=exp.episode(11)
        x=exp.feature(record,2,True)
        masked=exp.feature(record,2,False)
        self.assertEqual(len(x),len(masked))
        np.testing.assert_array_equal(x[:3],masked[:3])
        np.testing.assert_array_equal(masked[3:],np.zeros(4))
        np.testing.assert_array_equal(x[3:5],record["camera_actions"][1])
        np.testing.assert_array_equal(x[5:7],record["camera_actions"][2])

    def test_fit_and_held_out_evaluation(self):
        r=exp.run(out=None,train_episodes=3,validation_episodes=2,
                  test_episodes=2,steps=5)
        self.assertIn("learned_action",r["test"])
        self.assertIn("shuffled_action",r["test"])
        self.assertTrue(np.isfinite(r["test"]["learned_action"]["motion_mse"]))


if __name__=="__main__":
    unittest.main()
