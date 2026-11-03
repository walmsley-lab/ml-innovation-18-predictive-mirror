import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.frozen_rgb_campaign import run, paired_interval


class FrozenCampaignTests(unittest.TestCase):
    def test_identical_prediction_has_zero_paired_delta(self):
        frames = np.zeros((3, 2, 8, 8, 3), np.float32)
        self.assertEqual(paired_interval(frames, frames, frames, 2)["mean_delta_mse"], 0.)

    def test_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            out = run(folder, seeds=(1,), steps=1, context=3, horizon=2,
                      size=16, width=4, batch_size=2, train_count=3,
                      val_count=2, test_count=2)
            self.assertEqual(out["protocol"], "frozen-rgb-campaign-v1")
            self.assertEqual(set(out["runs"][0]["models"]),
                             {"nca", "cnn", "transport"})
            self.assertTrue((Path(folder)/"metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
