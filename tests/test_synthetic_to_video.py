import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.synthetic_to_video import run
from src.procedural_video import batch, sequence


class TransferTests(unittest.TestCase):
    def test_2d_curriculum_is_deterministic_and_observation_only(self):
        for stage in range(4):
            a = sequence(12, size=16, stage=stage)
            np.testing.assert_array_equal(a, sequence(12, size=16, stage=stage))
            self.assertEqual(a.shape, (9, 16, 16, 1))
            self.assertGreaterEqual(a.min(), 0.0)
            self.assertLessEqual(a.max(), 1.0)
        self.assertFalse(np.array_equal(sequence(12, size=16, stage=0),
                                        sequence(12, size=16, stage=3)))
        self.assertEqual(batch([1, 2], [0, 3], size=16).shape, (2, 9, 16, 16, 1))

    def test_same_initialization_zero_pretraining_gives_identical_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "webcam.npz"
            np.savez_compressed(path, frames=np.random.default_rng(8).random(
                (120, 16, 16, 1), dtype=np.float32))
            result = run(path, Path(directory) / "out", seed=4,
                         pretrain_steps=0, finetune_steps=2, batch_size=2)
            self.assertAlmostEqual(result["transfer_gain_mse"], 0.0, places=8)
            self.assertTrue((Path(directory) / "out" / "pretrained.npz").is_file())
            self.assertTrue((Path(directory) / "out" / "scratch.npz").is_file())

    def test_small_synthetic_pretraining_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "webcam.npz"
            np.savez_compressed(path, frames=np.random.default_rng(2).random(
                (120, 16, 16, 1), dtype=np.float32))
            result = run(path, Path(directory) / "out", seed=4,
                         pretrain_steps=4, finetune_steps=2, batch_size=2)
            self.assertEqual(result["pretrain_steps"], 4)
            self.assertIn("motion_region_mse", result["pretrained"])


if __name__ == "__main__":
    unittest.main()
