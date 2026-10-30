import unittest

from report import markdown, groups_of, feedback_svg


def row(seed, curriculum, mae, n=2):
    return {"architecture": "recurrent", "curriculum": curriculum,
            "loss_mode": "plain", "seed": seed, "test_mse": 0.02,
            "reappearance_location_mae_px": mae, "reappearance_count": n}


class EvidenceReportTests(unittest.TestCase):
    def test_source_based_report_and_seed_pairing(self):
        output = markdown([row(1, "staged", 4), row(1, "shuffled", 6)])
        self.assertIn("staged−shuffled MAE = -2.0000", output)
        self.assertIn("Exploratory engineering evidence", output)

    def test_invalid_runs_and_empty_events_fail_honestly(self):
        self.assertIn("NA", markdown([row(2, "staged", None, 0)]))
        with self.assertRaises(ValueError):
            groups_of([row(3, "staged", 4), row(3, "staged", 4)])
        with self.assertRaises(ValueError):
            groups_of([row(3, "staged", None, 1)])
        with self.assertRaises(ValueError):
            markdown([])

    def test_optional_figure_does_not_fabricate_occlusion(self):
        fixture = {regime: {label: {"mse": 0.02, "reappearance_location_mae_px": None}
                            for label in ("intact_trained", "feedback_free_trained", "intact_cut_at_test")}
                   for regime in ("moving_object_still_camera", "moving_object_moving_camera")}
        svg = feedback_svg(fixture)
        self.assertIn("NA", svg)
        self.assertIn("Separate units and scales", svg)
        self.assertNotIn("nan", svg)


if __name__ == "__main__":
    unittest.main()
