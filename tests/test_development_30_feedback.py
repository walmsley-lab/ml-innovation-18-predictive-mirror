import unittest
import numpy as np
from experiments import development_30_feedback as exp


class Development30Tests(unittest.TestCase):
    def test_feedback_is_real_not_a_decorative_channel(self):
        s=np.zeros((24,24));s[5,5]=1.;s[18,18]=1.
        left=exp.decode(s,prior=[5,5])
        right=exp.decode(s,prior=[18,18])
        self.assertLess(left[0],right[0])
        self.assertLess(left[1],right[1])

    def test_causal_prefix(self):
        frames,_,_=exp.world()
        other=exp.world(22)[0]
        for arm in ("feedback","no_feedback","shuffled"):
            a=exp.infer_series(frames[:6],arm,other[:6])
            b=exp.infer_series(frames,arm,other)
            for x,y in zip(a,b):
                np.testing.assert_allclose(x["slot"],y["slot"])
                np.testing.assert_allclose(x["pixel"],y["pixel"])

    def test_score_with_conflicting_control(self):
        report=exp.run(out=None,seeds=range(91000,91003),ticks=2)
        self.assertIn("shuffled",report["scores"])
        self.assertIn("no_feedback",report["scores"])
        self.assertIn("slot_visible_mae_px",report["scores"]["feedback"])
        with self.assertRaises(ValueError):
            exp.infer_series(exp.world()[0],"shuffled")


if __name__=="__main__":
    unittest.main()
