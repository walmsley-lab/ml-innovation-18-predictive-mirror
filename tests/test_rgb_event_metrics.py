"""Self-contained event metric contract checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.rgb_event_benchmark import run
from src.rgb_event_metrics import score
from src.rgb_motion_world import episode


class RGBEventMetricsTests(unittest.TestCase):
    def setUp(self):
        self.records = [episode(123, steps=9, size=24, balls=2, occlusion=True)]
        self.truth = self.records[0]["frames"][4:9][None]

    def test_perfect_forecast_scores_zero(self):
        result = score(self.truth, self.records, context=4, leads=(1,3,5))
        for values in result.values():
            self.assertAlmostEqual(values["mse"], 0)
            if values["foreground_mse"] is not None:
                self.assertAlmostEqual(values["foreground_mse"], 0)
            if values["visible_location_mae_px"] is not None:
                self.assertAlmostEqual(values["visible_location_mae_px"], 0)

    def test_missing_foreground_gets_penalty(self):
        missing = np.zeros_like(self.truth)
        result = score(missing, self.records, 4, (1,3,5))
        self.assertGreater(result["1"]["mse"], 0)
        visible_count = sum(v["visible_objects"] for v in result.values())
        missed = sum(v["missing_visible_predictions"] for v in result.values())
        self.assertGreater(visible_count, 0)
        self.assertEqual(missed, visible_count)

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            score(self.truth, self.records, 4, (6,))

    def test_runner_outputs_baselines_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"metrics.json"
            report = run(output=out, episodes=2, context=4, horizon=3,
                         size=24, balls=1, occlusion=False)
            self.assertTrue(out.exists())
            self.assertEqual(set(report["metrics"]), {"persistence","constant_velocity"})
            self.assertIn("foreground_mse", report["metrics"]["persistence"]["3"])


if __name__ == "__main__":
    unittest.main()
