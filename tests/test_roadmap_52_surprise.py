import unittest
import numpy as np
from experiments import roadmap_52_surprise as m


class SurpriseTests(unittest.TestCase):
    def test_future_value_does_not_revise_issued_prediction(self):
        obs=m.world(11)
        a,_=m.infer(obs)
        modified=obs.copy();modified[12:]+=100
        b,_=m.infer(modified)
        np.testing.assert_array_equal(a[:11],b[:11])

    def test_delay_control_and_zero_leak(self):
        obs=m.world(11,kind="both")
        for mode in m.MODES:
            p,err=m.infer(obs,mode)
            self.assertEqual(len(p),len(obs)-2)
            self.assertTrue(np.isfinite(err).all())

    def test_event_strata(self):
        r=m.run(None,seeds=(11,))
        self.assertEqual(len(r["rows"]),12)
        self.assertIn("recovery_next5_mse",r["rows"][0]["scores"])


if __name__=="__main__":
    unittest.main()
