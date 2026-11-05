import unittest
import numpy as np
from experiments import visual_memory_bridge as m


class VisualMemoryBridgeTests(unittest.TestCase):
    def test_prediction_scored_before_learning(self):
        prior=np.zeros((3,4,3))
        target=np.full_like(prior,.5)
        events=[[(prior,target),(prior,target)]]*3
        adaptive=m.train_events("recent_only",events,rate=.1,capacity=2)
        frozen=m.train_events("frozen",events,rate=.1,capacity=2)
        self.assertEqual(adaptive["updates"],12)
        self.assertEqual(frozen["updates"],0)
        self.assertGreater(adaptive["stages"][0]["start_mse"],0)
        # The first score is independent of its target's training effect.
        self.assertEqual(adaptive["stages"][0]["stage"],"A")
        self.assertLess(adaptive["stages"][-1]["end_mse"],frozen["stages"][-1]["end_mse"])

    def test_causal_replay_and_equal_updates(self):
        x=np.zeros((3,3,3))
        y=np.ones((3,3,3))*.2
        stream=[[(x,y) for _ in range(3)] for _ in range(3)]
        a=m.train_events("replay",stream,capacity=2)
        b=m.train_events("recent_only",stream,capacity=2)
        self.assertEqual(a["updates"],b["updates"])
        self.assertEqual(a["replayed"],8)
        self.assertLessEqual(a["memory_entries"],2)
        self.assertGreater(a["memory_bytes"],0)

    def test_gates_do_not_see_test(self):
        self.assertEqual(m.visual_gate([[.1,.2],[.1,.3],[.1,.1]],[.25,.25,.25]),1)
        self.assertEqual(m.visual_gate([[.1,.2],[.1,.2],[.1,.1]],[.25,.25,.25]),3)
        with self.assertRaises(ValueError):
            m.visual_gate([[.1]],[])

    def test_cpu_model_smoke(self):
        result=m.run(None,smoke=True,seeds=(11,),context=3,
                     length=2,size=20,width=2,ticks=1,capacity=2)
        self.assertEqual(len(result["rows"]),3)
        for row in result["rows"]:
            self.assertEqual(row["modes"]["recent_only"]["updates"],12)
            self.assertIn("known_action_mse",row["action_control"])


if __name__=="__main__":
    unittest.main()
