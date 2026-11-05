import unittest
import numpy as np
from experiments import roadmap_33_replay as m


class ReplayTests(unittest.TestCase):
    def test_prior_only_memory_and_budget(self):
        for mode in ("replay","recent_only"):
            result=m.learn(3,mode,capacity=5,stages=(5,7,4))
            self.assertEqual(result["gradient_updates"],2*result["total_examples"])
            self.assertLess(result["max_replay_index"],result["total_examples"])
            self.assertEqual(len(result["stages"]),3)
        self.assertGreater(m.learn(3,"replay",capacity=5,lr=.08,
                                   stages=(5,7,4))["replayed_updates"],0)

    def test_seeded_reproducibility_and_capacity(self):
        a=m.learn(13,"replay",capacity=4,stages=(5,8,4))
        b=m.learn(13,"replay",capacity=4,stages=(5,8,4))
        np.testing.assert_allclose(a["weights"],b["weights"])
        self.assertEqual(a["stages"],b["stages"])

    def test_both_tasks_preserved_as_separate_measures(self):
        report=m.run(None,seeds=(11,),capacity=4)
        for arm in ("replay","recent_only"):
            self.assertIn("A_forgetting_delta",report["test"][arm])
            self.assertIn("B_after_B",report["test"][arm])


if __name__=="__main__":
    unittest.main()
