import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sweep import run


class SweepResumeTests(unittest.TestCase):
    def test_complete_run_is_skipped_and_conflicting_config_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            args = (folder, [17], ["recurrent"], ["baseline"], 1, 2, 2, False)
            run(*args, checkpoint_every=1)
            original = (Path(folder) / "runs.json").read_bytes()
            with patch("sweep.train", side_effect=AssertionError("trained twice")):
                run(*args, resume=True, checkpoint_every=1)
            self.assertEqual(original, (Path(folder) / "runs.json").read_bytes())
            with self.assertRaises(FileExistsError):
                run(*args, checkpoint_every=1)
            with self.assertRaises(ValueError):
                run(folder, [17], ["recurrent"], ["baseline"], 2, 2, 2, False,
                    resume=True, checkpoint_every=1)

    def test_interrupted_matrix_recovers_completed_first_run(self):
        from sweep import train as original_train
        with tempfile.TemporaryDirectory() as folder:
            args = (folder, [18, 19], ["recurrent"], ["baseline"], 1, 2, 2, False)
            def stop_after_first(**kwargs):
                if kwargs["seed"] == 19:
                    raise RuntimeError("simulated interruption")
                return original_train(**kwargs)
            with patch("sweep.train", side_effect=stop_after_first):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run(*args, checkpoint_every=1)
            self.assertEqual(len(json.loads((Path(folder) / "runs.json").read_text())), 1)
            def expect_only_second(**kwargs):
                self.assertEqual(kwargs["seed"], 19)
                return original_train(**kwargs)
            with patch("sweep.train", side_effect=expect_only_second) as observed:
                rows = run(*args, resume=True, checkpoint_every=1)
            self.assertEqual(observed.call_count, 1)
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
