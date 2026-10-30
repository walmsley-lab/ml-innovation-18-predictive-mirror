import tempfile
import unittest
from pathlib import Path

from deploy.akash_preflight import render, save


DIGEST = ("ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror@sha256:"
          + "a" * 64)
SDL = """version: "2.0"
services:
  trainer:
    image: ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror:bootstrap-1
    env:
      - RUN_MODE=sweep
      - LOSS_MODE=contrast # observation only
      - CHECKPOINT_EVERY=25
      - SEED=42
      - STEPS=120
      - BATCH_SIZE=8
      - REQUIRE_GPU=1
    expose:
      - port: 8080
profiles:
  compute:
    trainer:
      resources:
        gpu:
          units: 1
  placement:
    akash:
      pricing:
        trainer:
          denom: uact
          amount: 2500 # max bid
"""


class AkashPreflightTests(unittest.TestCase):
    def test_generates_safe_three_step_gpu_smoke(self):
        result = render(SDL, DIGEST)
        self.assertIn("image: " + DIGEST, result)
        for value in ("RUN_MODE=train", "STEPS=3", "BATCH_SIZE=2",
                      "CHECKPOINT_EVERY=1", "LOSS_MODE=plain", "REQUIRE_GPU=1"):
            self.assertIn("- " + value, result)
        self.assertNotIn("bootstrap-1", result)
        self.assertNotIn("STEPS=120", result)

    def test_rejects_unpinned_and_oversized_or_ambiguous_deployments(self):
        for bad in ("ghcr.io/example:latest",
                    DIGEST[:-1], DIGEST.replace("@sha256:", ":sha-"),
                    "ghcr.io/another/model@sha256:" + "a" * 64):
            with self.assertRaises(ValueError):
                render(SDL, bad)
        with self.assertRaises(ValueError):
            render(SDL, DIGEST, steps=120)
        with self.assertRaises(ValueError):
            render(SDL, DIGEST, batch_size=8)
        with self.assertRaises(ValueError):
            render(SDL.replace("2500", "2501"), DIGEST)
        with self.assertRaises(ValueError):
            render(SDL.replace("      - STEPS=120\n", ""), DIGEST)
        with self.assertRaises(ValueError):
            render(SDL.replace("      - STEPS=120", "      - STEPS=120\n      - STEPS=99"), DIGEST)

    def test_respects_immutable_reviewed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "smoke.yaml"
            self.assertEqual(save(path, "safe"), path)
            self.assertEqual(save(path, "safe"), path)
            with self.assertRaises(FileExistsError):
                save(path, "different")


if __name__ == "__main__":
    unittest.main()
