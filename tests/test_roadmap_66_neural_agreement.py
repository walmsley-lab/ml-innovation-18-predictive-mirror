import tempfile
from pathlib import Path
import unittest
import numpy as np
from experiments import roadmap_66_neural_agreement as m


class NeuralAgreementTests(unittest.TestCase):
    def test_session_split_and_causal_lags(self):
        x,y,s=m.toy()
        train,val,test=m.splits(s)
        self.assertFalse(set(s[train]) & set(s[test]))
        self.assertFalse(set(s[val]) & set(s[test]))
        self.assertEqual(m.examples(x,y,train,2)[0].shape[0],
                         len(train)*(x.shape[1]-2))
        with self.assertRaises(ValueError):
            m.examples(x,y,train,0)

    def test_synthetic_not_measurement(self):
        r=m.run(None)
        self.assertEqual(r["source_type"],"synthetic")
        self.assertIsNone(r["noise_ceiling"])
        self.assertIn(r["selected_lag_frames"],m.LAGS)

    def test_missing_measurement_provenance_fails(self):
        with self.assertRaises(ValueError):
            m.run(None,path="missing.npz",metadata=None)


if __name__=="__main__":
    unittest.main()
