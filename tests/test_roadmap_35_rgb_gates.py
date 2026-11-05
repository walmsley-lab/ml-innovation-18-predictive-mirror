import unittest
import numpy as np
from experiments import roadmap_35_rgb_gates as m


class RGBGatingTests(unittest.TestCase):
    def test_gate_needs_two_validation_successes(self):
        self.assertEqual(m.gate(0,0,6,.8,1.),(0,1,False))
        self.assertEqual(m.gate(0,1,6,.8,1.),(1,0,True))
        self.assertEqual(m.gate(0,1,6,1.1,1.),(0,0,False))
        self.assertEqual(m.gate(0,0,1,.8,1.),(0,0,False))

    def test_independent_visual_domains(self):
        a=m.samples(31000,2,"motion",context=3,size=16)
        b=m.samples(91000,2,"motion",context=3,size=16)
        self.assertFalse(np.array_equal(a,b))
        self.assertEqual(a.shape,(2,4,16,16,3))
        with self.assertRaises(ValueError):
            m.samples(31000,2,"oracle",context=3,size=16)

    def test_matched_training_steps_and_test_scoring(self):
        rows=[m.learn(3,arm,steps=4,context=3,size=16,
                      channels=2,ticks=1,validate_every=2)
              for arm in m.ARMS]
        for r in rows:
            self.assertEqual(r["steps"],4)
            self.assertEqual(sum(r["stage_visits"].values()),4)
            self.assertIn("occlusion",r["test"])
            self.assertIn("constant_velocity",r["test_controls"]["motion"])


if __name__=="__main__":
    unittest.main()
