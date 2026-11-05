import unittest
import numpy as np
from experiments import roadmap_35_gates as m


class GateTests(unittest.TestCase):
    def test_two_validations_required_and_reset_after_failure(self):
        stage,streak,promoted=m.promote(0,0,10,.001)
        self.assertEqual((stage,streak,promoted),(0,1,False))
        self.assertEqual(m.promote(stage,streak,10,.001),(1,0,True))
        self.assertEqual(m.promote(0,1,10,1.),(0,0,False))
        self.assertEqual(m.promote(0,0,1,.001),(0,0,False))

    def test_identical_update_budget_and_disjoint_test(self):
        fitted=[m.fit(3,mode,steps=30,batch=3) for mode in m.MODES]
        for entry in fitted:
            self.assertEqual(entry["updates"],30)
            self.assertEqual(entry["examples_seen"],90)
            self.assertEqual(entry["replay_simple_updates"],3)
            self.assertEqual(sum(entry["stage_visits"].values()),30)
            self.assertIn("nonlinear",entry["test_mse_by_stage"])
        x1,_=m.world(31000,0)
        x2,_=m.world(91000,0)
        self.assertFalse(np.array_equal(x1,x2))

    def test_reproducible_gating_and_stage_indices(self):
        a=m.fit(5,"gated",steps=30,batch=4)
        b=m.fit(5,"gated",steps=30,batch=4)
        self.assertEqual(a["promotions"],b["promotions"])
        np.testing.assert_allclose(a["weights"],b["weights"])


if __name__=="__main__":
    unittest.main()
