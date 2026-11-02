"""Evidence grouping rejects improper ranking and invented measurements."""
import json
import tempfile
import unittest
from pathlib import Path
from experiments.evidence_report import collect, nested, run


def entry(name, source, representation="RGB", path="score"):
    return {
        "name": name, "technique": name, "representation": representation,
        "benchmark": "synthetic-v1", "metric": "MSE", "unit": "squared RGB",
        "direction": "lower", "horizon_frames": 10, "split": "test",
        "seed_protocol": "seeds 11 and 23",
        "checkpoint_selection": "validation-selected model",
        "source": str(source), "metric_path": path}


class EvidenceTests(unittest.TestCase):
    def test_nested_metric_validation(self):
        self.assertEqual(nested({"x": [{"score": .125}]}, "x.0.score"), .125)
        with self.assertRaises(ValueError):
            nested({"x": None}, "x")
        with self.assertRaises(ValueError):
            nested({"x": 1}, "missing")
        with self.assertRaises(ValueError):
            nested({"x": float("nan")}, "x")

    def test_group_only_same_protocol_and_verify_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "metrics.json"
            source.write_text(json.dumps({"score": .42, "other": .8}))
            rows = [entry("A", source), entry("B", source),
                    entry("Graph", source, "given 2D joints")]
            rows[1]["metric_path"] = "other"
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"entries": rows}))
            report = run(manifest, folder / "report")
            self.assertEqual(len(report["groups"]), 2)
            self.assertEqual(report["groups"][0]["entries"], ["A", "B"])
            self.assertTrue(report["groups"][0]["comparable"])
            self.assertFalse(report["groups"][1]["comparable"])
            self.assertEqual(len(report["evidence"][0]["source_sha256"]), 64)
            self.assertTrue((folder / "report" / "report.md").exists())
            self.assertNotIn("Graph |", (folder / "report" / "report.md").read_text().split(
                "## Group 2")[0])

    def test_missing_metrics_never_get_fabricated(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"entries": [
                entry("absent", folder / "missing.json")]}))
            with self.assertRaises(FileNotFoundError):
                collect(manifest)
            report = collect(manifest, allow_missing=True)
            self.assertEqual(report["evidence"], [])
            self.assertEqual(len(report["missing"]), 1)

    def test_bad_schema_and_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "m.json"
            source.write_text('{"score": 1}')
            e = entry("same", source)
            manifest = folder / "m-manifest.json"
            manifest.write_text(json.dumps({"entries": [e, e]}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collect(manifest)


if __name__ == "__main__":
    unittest.main()
