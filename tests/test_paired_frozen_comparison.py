import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.paired_frozen_comparison import generated,run,model_forecast


class PairedScoreboardTests(unittest.TestCase):
    def test_same_last_frame_different_future(self):
        records,prefix,truth=generated((71000,71001))
        self.assertEqual(prefix.shape,(4,8,32,32,3))
        np.testing.assert_array_equal(prefix[::2,-1],prefix[1::2,-1])
        self.assertGreater(float(np.abs(truth[::2]-truth[1::2]).mean()),0)

    def test_oracle_only_if_declared_as_such(self):
        with tempfile.TemporaryDirectory() as td:
            _,_,truth=generated((71000,71001))
            p=Path(td)/"oracle.npz"
            np.savez_compressed(p,predictions=truth.transpose(0,1,4,2,3))
            report=run(Path(td)/"report.json",(71000,71001),
                       ("oracle:declared_known_prior:btchw:"+str(p),))
            self.assertAlmostEqual(report["results"]["oracle"]["rgb_mse"],0.)
            self.assertNotIn("oracle",dict(report["fair_ranking_by_event_success"]))

    def test_reject_bad_input(self):
        with self.assertRaises(ValueError):
            generated((11,11))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"bad.npz"
            np.savez_compressed(p,predictions=np.zeros((1,1,3,8,8)))
            with self.assertRaises(ValueError):
                model_forecast(p,(4,10,32,32,3),"btchw")


if __name__=="__main__":
    unittest.main()
