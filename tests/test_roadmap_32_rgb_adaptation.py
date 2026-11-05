import unittest
import numpy as np
from experiments import roadmap_32_rgb_adaptation as m


class RGBAdaptationTests(unittest.TestCase):
    def test_regime_and_stream_alignment(self):
        frames,regimes=m.stream(11,period=4,context=3,size=24)
        self.assertEqual(frames.shape,(15,24,24,3))
        self.assertEqual(regimes.tolist(),["A"]*4+["B"]*4+["A"]*4)

    def test_future_suffix_cannot_change_prior_forecasts(self):
        params=m.initialize(4,size=24,context=3,ticks=1,
                            pretrain_steps=0,channels=2)
        frames,_=m.stream(11,period=4,context=3,size=24)
        initial=m.evaluate(params,frames,context=3,ticks=1,mode="online")
        modified=frames.copy()
        modified[8:]=0
        changed=m.evaluate(params,modified,context=3,ticks=1,mode="online")
        # Loss at step j uses frame j as its *target*. First changed
        # observation at index 8 affects errors from index 7 onward,
        # but no earlier committed forecast / scores.
        np.testing.assert_allclose(initial["rgb_mse"][:5],
                                   changed["rgb_mse"][:5],atol=1e-7)

    def test_small_rgb_smoke(self):
        report=m.run(None,seeds=(3,),period=4,context=3,size=24,
                     channels=2,ticks=1,pretrain_steps=0,rate=.01)
        self.assertIn("online",report["aggregate"])
        self.assertIn("frozen",report["aggregate"])
        self.assertEqual(len(report["per_seed"][0]["results"]["online"]["last_errors"]),12)


if __name__=="__main__":
    unittest.main()
