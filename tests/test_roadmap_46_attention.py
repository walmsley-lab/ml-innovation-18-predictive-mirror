import unittest
import numpy as np
from experiments import roadmap_46_attention as m


class AttentionTests(unittest.TestCase):
    def test_future_does_not_enter_selection(self):
        frames=np.zeros((5,24,24,3));frames[-1,3:8,2:7]=1
        self.assertEqual(m.choose(frames,"surprise"),
                         m.choose(frames.copy(),"surprise"))
        self.assertEqual(len(m.choose(frames)),3)

    def test_controls_equal_window_count(self):
        past,event=m.episode(91000)
        for arm in m.MODES:
            origin=m.choose(past,arm,count=2,width=6)
            self.assertEqual(len(origin),2)
            self.assertGreater(m.mask(origin,24,6).sum(),0)

    def test_independent_events(self):
        r=m.run(None,seeds=(91000,91001),domains=("linear",))
        self.assertEqual(len(r["rows"]),8)
        self.assertTrue(all(0<=x["event_recall"]<=1 for x in r["rows"]))


if __name__=="__main__":
    unittest.main()
