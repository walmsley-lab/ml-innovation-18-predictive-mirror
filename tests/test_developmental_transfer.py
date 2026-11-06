import tempfile
import unittest
from pathlib import Path

from experiments.developmental_transfer import run


class DevelopmentalTransferTests(unittest.TestCase):
    def test_matched_arms_and_forecast_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run(folder, seeds=(11,), pretrain_updates=3,
                         visual_updates=3, size=16, sheet_size=4)
            self.assertTrue(report["same_posttraining_sequences"])
            self.assertIn("forecast captured before", report["forecast_scoring"])
            self.assertEqual(len(report["results"]), 5)
            for row in report["results"]:
                self.assertEqual(len(row["checkpoints"]), 3)
                self.assertTrue(all(Path(path).exists()
                                    for path in row["checkpoints"]))
            frozen = next(x for x in report["results"]
                          if x["arm"] == "A_coherent_frozen")
            self.assertEqual(frozen["pre_to_post_connection_delta"], 0.)


if __name__ == "__main__":
    unittest.main()
