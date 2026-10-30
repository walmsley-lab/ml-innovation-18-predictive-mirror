import tempfile
import unittest
from pathlib import Path

from experiments.developmental_transfer import compare, evaluate_snapshots
from train import train


class DevelopmentalTransferTests(unittest.TestCase):
    def test_staged_and_shuffled_snapshots_evaluate_on_identical_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            ordered, shuffled = Path(root) / "ordered", Path(root) / "shuffled"
            train(seed=6, steps=4, batch_size=2, curriculum="staged",
                  save_milestones=True, output=str(ordered), checkpoint_every=1)
            train(seed=6, steps=4, batch_size=2, curriculum="shuffled",
                  save_milestones=True, output=str(shuffled), checkpoint_every=1)
            result = compare(ordered, shuffled, "recurrent", [70_000_000])
            self.assertEqual(len(result["staged"]), 4)
            self.assertEqual(len(result["shuffled"]), 4)
            for step, tasks in result["staged"].items():
                self.assertEqual(set(tasks), {"0", "1", "2", "3"})
                self.assertIn(step, result["shuffled"])

    def test_missing_snapshot_fails(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                evaluate_snapshots(root, "recurrent", [70_000_000])


if __name__ == "__main__":
    unittest.main()
