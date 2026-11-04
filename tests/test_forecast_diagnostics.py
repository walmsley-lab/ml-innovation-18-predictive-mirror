"""Collapse diagnostics must flag a dead forecast and pass a live one."""
import unittest

import numpy as np

from src import forecast_diagnostics as diag


class ForecastDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.leads = (1, 2, 3)
        rng = np.random.default_rng(0)
        # A healthy forecaster: per-scene structure, interior (unrailed) values.
        self.live = 0.2 + 0.3 * rng.random((4, 3, 8, 8, 3), dtype=np.float32)
        # A collapsed forecaster: every scene predicts the same all-zero frame.
        self.dead = np.zeros((4, 3, 8, 8, 3), dtype=np.float32)

    def test_clip_saturation_detects_railed_frames(self):
        dead = diag.clip_saturation(self.dead, self.leads)
        live = diag.clip_saturation(self.live, self.leads)
        for lead in map(str, self.leads):
            self.assertEqual(dead[lead], 1.0)
            self.assertEqual(live[lead], 0.0)

    def test_input_independence_detects_collapse(self):
        dead = diag.input_independence(self.dead, self.leads)
        live = diag.input_independence(self.live, self.leads)
        for lead in map(str, self.leads):
            self.assertEqual(dead[lead], 0.0)
            self.assertGreater(live[lead], 1e-2)

    def test_response_measures_known_perturbation(self):
        shifted = self.live + 0.1  # a known, uniform perturbation
        out = diag.response(self.live, {"shift": shifted,
                                        "identity": self.live}, self.leads)
        for lead in map(str, self.leads):
            self.assertAlmostEqual(out["shift"][lead], 0.1, places=5)
            self.assertEqual(out["identity"][lead], 0.0)

    def test_verdict_live_vs_dead(self):
        # A perturbation that moves a live forecast but not a dead (railed) one.
        live = diag.response(self.live, {"wipe": self.live + 0.2}, self.leads)
        dead = diag.response(self.dead, {"wipe": self.dead}, self.leads)
        self.assertTrue(all(
            v == "live" for v in diag.measurement_verdict(
                live, self.leads).values()))
        self.assertTrue(all(
            v == "dead" for v in diag.measurement_verdict(
                dead, self.leads).values()))

    def test_verdict_insensitive_when_forecast_is_healthy_but_unmoved(self):
        # Healthy, input-dependent forecast (self.live) that a tiny perturbation
        # barely moves: not collapsed, but the control can't certify it.
        responses = diag.response(self.live, {"tiny": self.live + 1e-6}, self.leads)
        sat = diag.clip_saturation(self.live, self.leads)
        indep = diag.input_independence(self.live, self.leads)
        verdict = diag.measurement_verdict(responses, self.leads,
                                           independence=indep, saturation=sat)
        self.assertEqual(set(verdict.values()), {"insensitive"})
        # The same tiny perturbation on a collapsed forecast is "dead".
        responses = diag.response(self.dead, {"tiny": self.dead}, self.leads)
        sat = diag.clip_saturation(self.dead, self.leads)
        indep = diag.input_independence(self.dead, self.leads)
        verdict = diag.measurement_verdict(responses, self.leads,
                                           independence=indep, saturation=sat)
        self.assertEqual(set(verdict.values()), {"dead"})

    def test_collapse_report_bundles_everything(self):
        report = diag.collapse_report(
            self.dead, self.leads, positive_control={"wipe": self.dead})
        self.assertIn("clip_rail_saturation", report)
        self.assertIn("input_independence", report)
        self.assertIn("positive_control_response", report)
        self.assertEqual(set(report["measurement_verdict"].values()), {"dead"})

    def test_rejects_bad_leads_and_shapes(self):
        with self.assertRaises(ValueError):
            diag.clip_saturation(self.live, (0,))
        with self.assertRaises(ValueError):
            diag.input_independence(self.live, (99,))
        with self.assertRaises(ValueError):
            diag.response(self.live, {"x": self.live[:, :2]}, self.leads)
        with self.assertRaises(ValueError):
            diag.clip_saturation(np.zeros((3, 3, 3)), (1,))


if __name__ == "__main__":
    unittest.main()
