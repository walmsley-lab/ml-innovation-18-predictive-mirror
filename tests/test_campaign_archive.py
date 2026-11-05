"""An archive is complete and verifiable or it is rejected."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.campaign_archive import archive, verify, sha256


class CampaignArchiveTests(unittest.TestCase):
    def test_missing_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root)/"source"
            source.mkdir()
            (source/"metrics.json").write_text(json.dumps({
                "protocol": "paired-diverse-rgb-curriculum-v1",
                "context": 4, "horizon": 2, "size": 20,
                "results": [{"model": "cnn", "seed": 11, "curriculum": "linear",
                             "selected_step": 1}]}))
            with self.assertRaises(FileNotFoundError):
                archive(source, Path(root)/"dest", "perceptual",
                        "a"*40, examples=False)

    def test_hash_validation_and_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root)/"source", Path(root)/"archive"
            source.mkdir()
            checkpoint = source/"cnn-seed11-linear-best.npz"
            np.savez_compressed(checkpoint, weights=np.ones(3))
            row = {"model": "cnn", "seed": 11, "curriculum": "linear",
                   "selected_step": 0, "checkpoint_sha256": sha256(checkpoint)}
            (source/"metrics.json").write_text(json.dumps({
                "protocol": "paired-diverse-rgb-curriculum-v1",
                "results": [row], "context": 4, "horizon": 2, "size": 20}))
            result = archive(source, output, "perceptual", "a"*40,
                             examples=False)
            self.assertEqual(result["per_seed_results"][0]["selected_step"], 0)
            self.assertEqual(verify(output), 2)
            (output/"checkpoints"/checkpoint.name).write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                verify(output)


if __name__ == "__main__":
    unittest.main()
