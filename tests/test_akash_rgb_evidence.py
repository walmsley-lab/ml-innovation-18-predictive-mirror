"""Offline checks for the Akash RGB research plan; no paid calls or JAX required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import akash_rgb_evidence as evidence
from experiments import developmental_evidence
from deploy import gpu_campaign


class TestAkashRGBEvidence(unittest.TestCase):
    def sample(self):
        return json.loads(Path("experiments/akash-rgb-smoke.json").read_text())

    def test_exact_condition_accounting(self):
        self.assertEqual(evidence.validate(self.sample()), 2)
        full = json.loads(Path("experiments/akash-rgb-full.json").read_text())
        self.assertEqual(evidence.validate(full), 18)
        self.assertEqual(len(list(evidence.jobs(full))), 18)
        self.assertEqual(gpu_campaign.load_manifest("experiments/akash-rgb-full.json")[0], 18)
        developmental = json.loads(Path("experiments/akash-developmental-pilot.json").read_text())
        self.assertEqual(developmental_evidence.validate(developmental), 20)
        self.assertEqual(len(list(developmental_evidence.jobs(developmental))), 20)
        self.assertEqual(gpu_campaign.load_manifest("experiments/akash-developmental-pilot.json")[0], 20)

    def test_expanded_world_and_resolution_manifests(self):
        for name,expected in (
                ("akash-rgb-architectures.json", 12),
                ("akash-rgb-physics.json", 18),
                ("akash-rgb-48px.json", 9),
                ("akash-rgb-64px.json", 6),
                ("akash-rgb-occlusion-all.json", 18)):
            path = "experiments/" + name
            cfg = json.loads(Path(path).read_text())
            with self.subTest(name=name):
                self.assertEqual(evidence.validate(cfg), expected)
                self.assertEqual(len(list(evidence.jobs(cfg))), expected)
                self.assertEqual(gpu_campaign.load_manifest(path)[0], expected)
        scale = json.loads(Path("experiments/akash-rgb-64px.json").read_text())
        self.assertIn("--train-domains", next(iter(evidence.jobs(scale)))[1])
        scale["occlusion_models"] = ["cnn"]
        with self.assertRaises(ValueError):
            evidence.validate(scale)
        scale["occlusion_models"] = []
        scale["train_domains"].append("alien-physics")
        with self.assertRaises(ValueError):
            evidence.validate(scale)

    def test_rejects_bad_configuration(self):
        for key, value in (("size", 64), ("kind", "legacy"),
                           ("steps", 0), ("models", ["not_a_model"]),
                           ("seeds", [11, 11]), ("context", 2)):
            cfg = self.sample()
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.validate(cfg)

    def test_paid_run_limit_blocks_before_execution(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            cfg.write_text(json.dumps(self.sample()))
            with patch.object(evidence.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    evidence.run(cfg, Path(d) / "output", max_runs=1, require_gpu=False)
                run.assert_not_called()

    def test_failed_condition_keeps_incremental_log_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            cfg.write_text(json.dumps(self.sample()))
            fake = type("Result", (), {"returncode": 7})()
            with patch.object(evidence.subprocess, "run", return_value=fake):
                with self.assertRaises(RuntimeError):
                    evidence.run(cfg, Path(d) / "output", max_runs=2, require_gpu=False)
            report = json.loads((Path(d) / "output" / "evidence.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["conditions"]), 1)
            self.assertEqual(report["conditions"][0]["exit_code"], 7)
            self.assertTrue((Path(d) / "output" / report["conditions"][0]["log"]).is_file())


if __name__ == "__main__":
    unittest.main()
