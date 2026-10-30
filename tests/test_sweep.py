import unittest

from sweep import aggregate


class SweepTests(unittest.TestCase):
    def test_aggregation_preserves_individual_seed_and_handles_missing_events(self):
        sample = [
            {"architecture": "recurrent", "curriculum": "staged", "seed": 1,
             "reappearance_location_mae_px": 3.0},
            {"architecture": "recurrent", "curriculum": "staged", "seed": 2,
             "reappearance_location_mae_px": 5.0},
            {"architecture": "multisheet", "curriculum": "staged", "seed": 1,
             "reappearance_location_mae_px": None},
        ]
        result = aggregate(sample)
        self.assertEqual(result["recurrent:staged:plain"]["seeds"], [1, 2])
        self.assertEqual(result["recurrent:staged:plain"]["moving_camera_reappearance_mae_mean_px"], 4.0)
        self.assertIsNone(result["multisheet:staged:plain"]["moving_camera_reappearance_mae_mean_px"])


if __name__ == "__main__":
    unittest.main()
