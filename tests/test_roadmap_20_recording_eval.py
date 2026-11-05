import tempfile
from pathlib import Path
import unittest
import numpy as np
from experiments import roadmap_20_recording_eval as m


class RecordingEvalTests(unittest.TestCase):
    def test_invalid_time_and_reused_session(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"a.npz"
            np.savez(path,frames=np.zeros((9,16,16,3)),
                     timestamps_ns=np.arange(9,dtype=np.int64)*100000000)
            with self.assertRaises(ValueError):
                m.run(None,path,path,"cnn",smoke=True,width=2)
            bad=Path(d)/"bad.npz"
            np.savez(bad,frames=np.zeros((9,16,16,3)),timestamps_ns=np.zeros(9,dtype=int))
            with self.assertRaises(ValueError):m.session(bad)

    def test_offline_issued_before_future(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/"cal.npz",Path(d)/"test.npz"
            for i,p in enumerate((a,b)):
                x=np.zeros((10,16,16,3),np.float32)
                x[:,4+i,4+i]=.4
                np.savez(p,frames=x,timestamps_ns=np.arange(10)*110000000)
            r=m.run(None,a,b,"cnn",smoke=True,context=3,lead=2,width=2)
            self.assertEqual(r["test"]["paired_targets"],6)
            self.assertEqual(r["test"]["actual_horizon_ms_median"],220.)
            self.assertIn("persistence",r["test"]["scores_mse"])


if __name__=="__main__":
    unittest.main()
