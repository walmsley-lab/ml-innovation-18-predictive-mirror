import unittest
import numpy as np
from src.active_gaze_comparison import paired_ci,compare


class ActiveGazeComparisonTests(unittest.TestCase):
    def test_identical_series_have_zero_delta(self):
        values=np.array([1.,2.,3.])
        result=paired_ci(values,values)
        self.assertEqual(result["mean_difference"],0)
        self.assertEqual(result["episode_bootstrap_95pct"],[0.,0.])

    def test_paired_same_world_controls(self):
        report=compare(seeds=(3,),episodes=4,context=5,horizon=3)
        self.assertEqual(set(report["metrics"]),{"fixed","random","active"})
        self.assertEqual(report["metrics"]["active"]["episodes"],4)
        self.assertEqual(report["paired_comparisons"]["active_minus_fixed"]["pairs"],4)
        for value in report["metrics"].values():
            self.assertGreaterEqual(value["visible_observation_fraction"],0)
            self.assertLessEqual(value["visible_observation_fraction"],1)

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            compare(episodes=0)


if __name__=="__main__":
    unittest.main()
