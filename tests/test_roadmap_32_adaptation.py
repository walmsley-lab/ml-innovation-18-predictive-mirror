import unittest
import numpy as np
from experiments import roadmap_32_adaptation as m


class AdaptationTests(unittest.TestCase):
    def test_scored_before_label_update(self):
        x=np.array([[1.,0.,0.],[1.,0.,0.]])
        y=np.array([2.,2.])
        pred,final=m.prequential(np.zeros(3),x,y,lr=.5)
        self.assertEqual(pred[0],0.)
        self.assertGreater(pred[1],0.)
        self.assertGreater(final[0],pred[1])
        np.testing.assert_array_equal(m.prequential(np.zeros(3),x,y,"frozen")[0],
                                      np.zeros(2))

    def test_future_target_changes_do_not_change_past_forecasts(self):
        x,y,_=m.samples(91000)
        initial=m.initialize(31000,1)
        original,_=m.prequential(initial,x,y,"adaptive")
        altered=y.copy(); altered[30:]+=1000
        changed,_=m.prequential(initial,x,altered,"adaptive")
        np.testing.assert_allclose(original[:31],changed[:31])

    def test_shift_and_return_scored_independently(self):
        r=m.run(None,test_seeds=(91000,),period=20)
        self.assertIn("new_regime",r["test"]["adaptive"])
        self.assertIn("return_last_ten",r["test"]["frozen"])
        self.assertIn(r["selected_learning_rate"],(.02,.06,.12))


if __name__=="__main__":
    unittest.main()
