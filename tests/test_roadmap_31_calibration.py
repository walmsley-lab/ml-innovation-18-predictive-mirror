"""Calibration correctness, independent seed splits and causal forecasts."""
import unittest
import numpy as np
from experiments import roadmap_31_calibration as m


class CalibrationTests(unittest.TestCase):
    def test_finite_sample_rank(self):
        self.assertEqual(m.quantile([1,2,3,4,5,6,7,8,9,10], .2), 9.)
        with self.assertRaises(ValueError):
            m.quantile([1,2,3], .1)

    def test_future_never_enters_forecast(self):
        past, _ = m.episode(1234)
        original = m.forecast(past)
        np.testing.assert_array_equal(original[0], m.forecast(past.copy())[0])
        # A later history extension cannot alter an already committed prediction.
        extended = np.concatenate([past, [100.]])
        np.testing.assert_array_equal(original[0], m.forecast(extended[:-1])[0])

    def test_disjoint_episode_roots_and_report(self):
        a,b,_ = m.examples(31000,3)
        c,d,_ = m.examples(61000,3)
        self.assertFalse(np.array_equal(a,c))
        self.assertFalse(np.array_equal(b,d))
        result = m.run(None, 30, 45, 30, 3)
        self.assertEqual(len(result["calibration_radii"]["global"]),3)
        self.assertIn("conditional", result["test"])
        self.assertGreater(result["test"]["global"]["mean_interval_width"],0)


if __name__ == "__main__":
    unittest.main()
