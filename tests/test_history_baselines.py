"""RGB-only history methods can distinguish opposing histories."""
import tempfile
import unittest
import numpy as np
from src import paired_occlusion_histories as world
from src import history_baselines as history
from experiments.history_baseline_eval import run


class HistoryBaselineTests(unittest.TestCase):
    def test_detect_and_estimate_from_rgb_without_palette(self):
        rec = world.pair(11)[0]
        observed = rec["frames"][:8]
        info = history.estimate(observed)
        self.assertIsNotNone(info)
        self.assertGreater(info["velocity"][0], 0)
        self.assertGreater(info["radius"], 0.5)

    def test_history_direction_matters_when_last_frame_is_identical(self):
        a,b = world.pair(23)
        prefix_a, prefix_b = a["frames"][:8], b["frames"][:8]
        np.testing.assert_array_equal(prefix_a[-1],prefix_b[-1])
        for method in (history.history_linear,history.kalman_history):
            p,q = method(prefix_a),method(prefix_b)
            self.assertEqual(p.shape,(10,32,32,3))
            self.assertGreater(float(np.mean(np.abs(p-q))), 0)

    def test_history_beats_persistence_on_simple_straight_task(self):
        a,b = world.pair(11)
        contexts=np.stack([a["frames"][:8],b["frames"][:8]])
        truth=np.stack([a["frames"][8:],b["frames"][8:]])
        old=np.repeat(contexts[:,-1:],10,axis=1)
        new=np.stack([history.history_linear(x) for x in contexts])
        self.assertLess(np.mean((new-truth)**2),np.mean((old-truth)**2))

    def test_no_detection_falls_back_to_persistence(self):
        frames=np.zeros((8,32,32,3),np.float32)
        for method in (history.history_linear,history.kalman_history):
            pred=method(frames)
            np.testing.assert_array_equal(pred,np.zeros((10,32,32,3)))

    def test_eval_declares_reference_tier_and_pair_units(self):
        with tempfile.TemporaryDirectory() as d:
            report=run(seeds=(11,23),out=d+"/result.json")
            self.assertEqual(report["models"]["known_prior_linear_reference"]["input_tier"],"declared_known_prior")
            self.assertEqual(len(report["models"]["rgb_history_kalman"]["pair_success"]),2)
            self.assertEqual(report["models"]["rgb_history_linear"]["input_tier"],"rgb_only")


if __name__=="__main__":
    unittest.main()
