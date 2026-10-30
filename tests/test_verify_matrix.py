import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy.verify_matrix import check
from sweep import run


class IntegrationGateTests(unittest.TestCase):
    def test_real_one_cell_run_and_missing_artifact_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            run(root, [3], ["recurrent"], ["baseline"], 1, 2, 1, False, checkpoint_every=1)
            report = check(root, [3], ["recurrent"], ["baseline"])
            self.assertEqual(report["n"], 1)
            with self.assertRaises(ValueError):
                check(root, [3, 5], ["recurrent"], ["baseline"])
            (Path(root) / "recurrent-baseline-seed3" / "checkpoint.npz").unlink()
            with self.assertRaises(ValueError):
                check(root, [3], ["recurrent"], ["baseline"])


if __name__ == "__main__":
    unittest.main()
