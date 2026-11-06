import tempfile
import unittest
from pathlib import Path

from experiments.developmental_training import run


class DevelopmentalTrainingTests(unittest.TestCase):
    def test_matched_run_saves_persistent_stage_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(folder, size=16, sheet_size=4, stage_frames=3)
            self.assertEqual(result["total_updates"], 9)
            for arm in result["arms"].values():
                self.assertEqual(len(arm["checkpoint_stages"]), 3)
                self.assertTrue(all(Path(path).exists()
                                    for path in arm["checkpoint_stages"]))
                self.assertGreater(arm["final_connection_locality"], 0.)
            self.assertIn("developmental", result["evaluation"])
            self.assertIn("no_wave", result["evaluation"])


if __name__ == "__main__":
    unittest.main()
