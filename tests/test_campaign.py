import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from campaign import plan, execute


MINI = {"seeds": [3], "architectures": ["recurrent"], "curricula": ["baseline"],
        "steps": 1, "batch_size": 2, "episodes": 1, "checkpoint_every": 1,
        "loss_mode": "plain", "primary_metric": "reappearance_location_mae_px",
        "replay_fraction": 0.10}


class CampaignTests(unittest.TestCase):
    def test_readonly_plan_and_budget_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "campaign"
            self.assertEqual(plan(MINI)["runs"], 1)
            with patch("campaign.run", side_effect=AssertionError("Training must not start")):
                with self.assertRaises(ValueError):
                    execute(MINI, output, 0)
            self.assertFalse(output.exists())

    def test_real_run_resume_and_config_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "campaign"
            self.assertEqual(len(execute(MINI, output, 1)), 1)
            self.assertEqual(len(execute(MINI, output, 1)), 1)
            self.assertEqual(json.loads((output / "campaign.json").read_text())["configuration"], MINI)
            with self.assertRaises(ValueError):
                execute({**MINI, "steps": 2}, output, 1)

    def test_reject_changed_primary_metric_or_duplicates(self):
        for invalid in ({**MINI, "primary_metric": "mse"},
                        {**MINI, "seeds": [3, 3]},
                        {**MINI, "curricula": ["staged", "staged"]}):
            with self.assertRaises(ValueError):
                plan(invalid)


if __name__ == "__main__":
    unittest.main()
