import tempfile
import unittest
import numpy as np
from src import multifuture_motion as m
from experiments.multifuture_calibration import run


class MultiFutureTests(unittest.TestCase):
    def test_identical_history_different_future(self):
        a=m.episode(11,0,context=5,horizon=4)
        b=m.episode(11,1,context=5,horizon=4)
        np.testing.assert_array_equal(a["frames"][:5],b["frames"][:5])
        self.assertGreater(np.max(abs(a["positions"][5:]-b["positions"][5:])),0)

    def test_proper_score_penalizes_overdispersed_and_wrong(self):
        truth=np.zeros((3,2))
        good=np.stack([truth,truth])
        bad=np.stack([truth+10,truth+20])
        weights=(.5,.5)
        self.assertLess(m.energy_score(good,truth,weights),
                        m.energy_score(bad,truth,weights))
        with self.assertRaises(ValueError):
            m.predict(np.ones((4,32,32,3)),4,probabilities=(1,1))

    def test_calibration_and_heldout_batch(self):
        with tempfile.TemporaryDirectory() as root:
            result=run(root+"/metrics.json",train_pairs=3,
                       test_pairs=3,horizon=3)
            self.assertAlmostEqual(result["branch_probabilities"][0],.5)
            self.assertEqual(len(result["rows"]),6)
            self.assertIn("energy_score",result["aggregate"])


if __name__=="__main__":
    unittest.main()
