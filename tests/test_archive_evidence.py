"""Portable evidence archive: hash verification does not imply reproduced science."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.archive_evidence import archive, verify


class EvidenceBundleTests(unittest.TestCase):
    def test_archives_and_verifies_without_promoting_claims(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metrics = root/"metrics.json"
            metrics.write_text(json.dumps({"mse": 0.02}))
            checkpoint = root/"weights.npz"
            checkpoint.write_bytes(b"example checkpoint")
            bundle = root/"bundle"
            report = archive(metrics, [checkpoint], bundle, command="python run.py",
                             selection="minimum validation loss")
            self.assertEqual(len(report["artifacts"]), 2)
            self.assertIn("NOT certified", report["attestation"])
            checked = verify(bundle)
            self.assertEqual(checked["file_count"], 2)
            self.assertFalse(checked["independent_reproduction_verified"])
            with self.assertRaises(FileExistsError):
                archive(metrics, [], bundle)

    def test_tampering_and_missing_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/"score.json"
            source.write_text('{"result": 1}')
            bundle = root/"bundle"
            archive(source, out=bundle)
            artifact = next((bundle/"artifacts").iterdir())
            artifact.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify(bundle)
            artifact.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                verify(bundle)

    def test_manifest_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/"score.json"
            source.write_text('{"result": 1}')
            bundle = root/"bundle"
            archive(source, out=bundle)
            manifest = bundle/"manifest.json"
            record = json.loads(manifest.read_text())
            record["artifacts"][0]["path"] = "../score.json"
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verify(bundle)


if __name__ == "__main__":
    unittest.main()
