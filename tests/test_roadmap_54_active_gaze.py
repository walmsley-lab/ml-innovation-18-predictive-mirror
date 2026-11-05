import unittest
import numpy as np
from experiments import roadmap_54_active_gaze as m


class Active2DGazeTests(unittest.TestCase):
    def test_belief_moves_and_remains_normalized(self):
        a=np.full((8,8),1/64)
        b=m.propagate(a)
        self.assertAlmostEqual(b.sum(),1.)
        visible,hit=m.observe(b,(0,0),np.array([1,1]))
        self.assertTrue(hit)
        self.assertEqual(visible[1,1],1.)
        hidden,hit=m.observe(b,(0,0),np.array([7,7]))
        self.assertFalse(hit)
        self.assertAlmostEqual(hidden.sum(),1.)

    def test_equal_sensing_and_no_oracle_selection(self):
        fixed=m.fixed_training(range(31000,31002))
        results=[m.episode(91000,policy,fixed,steps=5)
                 for policy in m.POLICIES]
        self.assertEqual({r["observations"] for r in results},{5})
        self.assertEqual(m.choose(np.full((8,8),1/64)),
                         m.choose(np.full((8,8),1/64)))

    def test_report(self):
        self.assertEqual(len(m.run(None,seeds=(91000,))["rows"]),3)


if __name__=="__main__":
    unittest.main()
