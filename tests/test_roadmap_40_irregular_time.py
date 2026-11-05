import unittest
import numpy as np
from experiments import roadmap_40_irregular_time as m


class IrregularTimeTests(unittest.TestCase):
    def test_causal_prefix_and_known_duration(self):
        r=m.episode(11,steps=9,domain="dropped")
        x,_=m.feature(r,4,True)
        altered={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in r.items()}
        altered["frames"][5:]=1.
        altered["positions_evaluation_only"][5:]=999.
        np.testing.assert_allclose(x,m.feature(altered,4,True)[0])
        self.assertEqual(len(x),5)

    def test_schedule_sensitive_controls(self):
        r=m.episode(13,steps=9,domain="jitter")
        aware,_=m.feature(r,3,True)
        blind,_=m.feature(r,3,False)
        np.testing.assert_allclose(aware[:3],blind[:3])
        np.testing.assert_array_equal(blind[3:],np.zeros(2))
        self.assertGreater(r["timestamps"][-1],0)

    def test_train_validation_test(self):
        r=m.run(None,train_count=3,validation_count=2,test_count=2)
        self.assertEqual(set(r["test"]),set(m.DOMAINS))
        self.assertIn("scheduled_velocity",r["test"]["dropped"])
        self.assertTrue(np.isfinite(r["test"]["regular"]["time_aware"]
                                     ["displacement_mse_px2"]))


if __name__=="__main__":
    unittest.main()
