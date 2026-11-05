import tempfile
import unittest
import numpy as np
from src import paired_occlusion_histories as paired
from experiments.paired_occlusion_histories import run,evaluate


class PairedHistoryTests(unittest.TestCase):
    def test_identical_final_different_history_and_future(self):
        for seed in (11,23,41):
            a,b=paired.pair(seed)
            np.testing.assert_array_equal(a["frames"][7],b["frames"][7])
            self.assertFalse(np.array_equal(a["frames"][:7],b["frames"][:7]))
            self.assertFalse(np.array_equal(a["frames"][8:],b["frames"][8:]))
            self.assertEqual(a["events"][0]["kind"],"barrier_return")
            self.assertEqual(b["events"][0]["kind"],"barrier_return")

    def test_perfect_contrast_and_reset(self):
        records=list(paired.pair(11))
        perfect=lambda x:np.stack([r["frames"][8:] for r in records])
        result=evaluate(perfect,records,8,10)
        self.assertAlmostEqual(result["intact"]["future_contrast_mse"],0)
        original=np.stack([r["frames"][:8] for r in records])
        controls=paired.controls(original)
        np.testing.assert_array_equal(controls["reset"][0],controls["reset"][1])
        np.testing.assert_array_equal(controls["shuffled"][:,-1],original[:,-1])

    def test_simple_history_baseline_beats_last_frame(self):
        records=paired.pair(11)
        motion=np.stack([paired.linear_history_forecast(
            r["frames"][:8]) for r in records])
        static=np.stack([np.repeat(r["frames"][7:8],10,axis=0)
                         for r in records])
        true=np.stack([r["frames"][8:] for r in records])
        self.assertLess(float(np.mean((motion-true)**2)),
                        float(np.mean((static-true)**2)))

    def test_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            result=run(directory+"/metrics.json",seeds=(11,))
            self.assertEqual(result["predictor"],"persistence")
            self.assertEqual(result["positive_control"]["barrier_events"],2)


if __name__=="__main__":
    unittest.main()
