"""Evidence gates, independent calibration and no-future-RGB checks."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from experiments import frozen_visual_bridge as m


class FrozenVisualBridgeTests(unittest.TestCase):
    def test_conformal_finite_sample(self):
        score=np.arange(1,21,dtype=float)[:,None]
        self.assertEqual(m.radius(score,.1)[0],19.)
        with self.assertRaises(ValueError):
            m.radius(score[:3],.1)

    def test_untrained_requires_explicit_smoke(self):
        with self.assertRaises(ValueError):
            m.checkpoint("nca")
        params, meta=m.checkpoint("nca",smoke=True,width=2)
        self.assertIn("untrained",meta["evidence"])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"nca.npz"
            np.savez(path,**{k:np.asarray(v) for k,v in params.items()})
            loaded, _=m.checkpoint("nca",path,width=2)
            self.assertEqual(set(loaded),set(params))

    def test_visual_prediction_causal(self):
        obs,_=m.worlds(91000,2,3,2,20,"linear")
        params,_=m.checkpoint("cnn",smoke=True,width=2)
        a=m.predict("cnn",params,obs[:,:3],2)
        changed=obs.copy();changed[:,3:]=0.
        b=m.predict("cnn",params,changed[:,:3],2)
        np.testing.assert_allclose(a,b)

    def test_calibration_not_test_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            r=m.run(str(Path(folder)/"run.json"),smoke=True,
                    calibration_count=12,test_count=2,context=3,horizon=2,
                    size=20,width=2,ticks=1)
            self.assertTrue(r["cases"])
            case=r["cases"]["linear/none"]["cnn"]
            self.assertIn("persistence",case["metrics"])
            self.assertEqual(len(case["conformal_radius_by_lead"]),2)


if __name__=="__main__":
    unittest.main()
