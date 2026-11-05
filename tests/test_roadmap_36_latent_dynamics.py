import tempfile
import unittest
import numpy as np
from experiments import roadmap_36_latent_dynamics as m


class LatentDynamicsTests(unittest.TestCase):
    def test_rank_budget_and_determinism(self):
        frames=m.scenes(31000,4,3,2,20)
        fitted=m.fit(frames,2,.1)
        self.assertEqual(fitted["basis"].shape[1],2)
        self.assertEqual(m.predict(fitted,frames[:,:3],2).shape,(4,2,20,20,3))
        np.testing.assert_allclose(m.predict(fitted,frames[:,:3],2),
                                   m.predict(m.fit(frames,2,.1),frames[:,:3],2))

    def test_test_suffix_never_enters_predictions(self):
        frames=m.scenes(91000,3,3,2,20)
        fitted=m.fit(m.scenes(31000,4,3,2,20),2)
        baseline=m.predict(fitted,frames[:,:3],2)
        modified=frames.copy();modified[:,3:]=1.
        np.testing.assert_array_equal(baseline,m.predict(fitted,modified[:,:3],2))

    def test_small_scientific_protocol(self):
        with tempfile.TemporaryDirectory() as p:
            result=m.run(p,train_episodes=4,valid_episodes=2,test_episodes=2,
                         context=3,horizon=2,size=20)
            self.assertIn("constant_velocity",result["cases"]["combined"])
            self.assertIsNotNone(result["checkpoint_sha256"])
            self.assertTrue(result["validation_selected_mse"]>=0)


if __name__=="__main__":
    unittest.main()
