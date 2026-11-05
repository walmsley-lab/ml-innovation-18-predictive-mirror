import unittest
import numpy as np
from experiments import development_29_active_sampling as exp


class Development29Tests(unittest.TestCase):
    def test_probability_and_no_oracle_action(self):
        b=np.full(exp.WIDTH,1/exp.WIDTH)
        x=exp.choose(b)
        self.assertIn(x,exp.WINDOWS)
        self.assertAlmostEqual(exp.transition(b).sum(),1.)
        for sighting in (None,x):
            posterior=exp.update(b,x,sighting)
            self.assertAlmostEqual(posterior.sum(),1.)
            self.assertTrue(np.isfinite(exp.entropy(posterior)))

    def test_equal_number_of_sensors_and_frozen_train_choice(self):
        fixed=exp.select_fixed(range(31000,31003))
        a=exp.episode(91000,"active",fixed,8)
        f=exp.episode(91000,"fixed",fixed,8)
        r=exp.episode(91000,"random",fixed,8)
        for p in (a,f,r):
            self.assertGreaterEqual(p["sighting"],0)
            self.assertLessEqual(p["sighting"],1)
            self.assertGreaterEqual(p["gaze_movement"],0)

    def test_small_held_out_report(self):
        result=exp.run(out=None,train_seeds=range(31000,31003),
                       test_seeds=range(91000,91003))
        self.assertIn("active",result["test"])
        self.assertIn("fixed",result["test"])
        self.assertIn("random",result["test"])


if __name__=="__main__":
    unittest.main()
