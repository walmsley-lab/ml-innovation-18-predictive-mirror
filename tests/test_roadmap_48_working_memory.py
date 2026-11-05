import unittest
import numpy as np
from experiments import roadmap_48_working_memory as m


class WorkingMemoryTests(unittest.TestCase):
    def test_capacity_and_missing_not_oracle(self):
        obs={i:np.array([float(i+5),10.]) for i in range(4)}
        for arm in ("recency","novelty","random"):
            bank=m.update({},obs,3,2,arm)
            self.assertEqual(len(bank),2)
            self.assertIsNone(m.predict(bank,100,4))

    def test_causal_update_past_only(self):
        frame=np.zeros((32,32,3))
        frame[4:7,4:7]=m.PALETTE[0]
        self.assertIn(0,m.detect(frame))
        bank=m.update({},m.detect(frame),0,2)
        self.assertTrue(np.isfinite(m.predict(bank,0,2)).all())

    def test_report(self):
        r=m.run(None,seeds=(91000,),steps=10)
        self.assertEqual(len(r["rows"]),3)
        self.assertTrue(all(x["hidden_events"]>=x["scorable_hidden"] for x in r["rows"]))


if __name__=="__main__":
    unittest.main()
