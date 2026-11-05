import unittest
import numpy as np
from experiments import development_25_completion as exp


class Development25Tests(unittest.TestCase):
    def test_occluder_and_training_target_are_observed_colors(self):
        clean, truth = exp.silhouette(31000)
        partial = exp.obscure(clean, 31000)
        self.assertGreater(np.sum(partial[...,2] > .6), 0)
        self.assertTrue(np.any(truth))
        self.assertEqual(exp.features(partial).shape[0], 2*16*16+1)
        np.testing.assert_allclose(exp.obscure(clean,0,full=True)[...,2], .85,
                                   atol=1e-6)

    def test_small_ridge_and_unseen_test(self):
        w=exp.fit(range(31000,31012))
        current,_=exp.silhouette(91000)
        predicted=exp.predict(w, exp.obscure(current,91000))
        self.assertEqual(predicted.shape,(16,16))
        self.assertTrue(np.isfinite(predicted).all())
        r=exp.run(out=None,train_count=12,test_count=3)
        self.assertIn("partial",r["scores"])
        self.assertIn("fully_hidden",r["scores"])

    def test_current_frame_only(self):
        w=exp.fit(range(31000,31008))
        clean,_=exp.silhouette(91020)
        frame=exp.obscure(clean,91020)
        np.testing.assert_array_equal(exp.predict(w,frame),exp.predict(w,frame.copy()))


if __name__=="__main__":
    unittest.main()
