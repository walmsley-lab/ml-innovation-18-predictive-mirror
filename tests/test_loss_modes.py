import tempfile
import unittest
from pathlib import Path

import jax

from src import model, feedforward, multisheet
from src.checkpoint import load
from src.world import batch
from train import train


class LossModeTests(unittest.TestCase):
    def test_observation_only_contrast_is_distinct_for_every_architecture(self):
        x, a = batch([13, 14], steps=4)
        for module in (model, feedforward, multisheet):
            params = module.init(jax.random.PRNGKey(9))
            self.assertGreater(float(module.loss(params, x, a, "contrast")),
                               float(module.loss(params, x, a, "plain")))

    def test_checkpoint_rejects_training_objective_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            train(seed=3, steps=2, batch_size=2, output=folder, loss_mode="contrast")
            with self.assertRaises(ValueError):
                load(Path(folder) / "checkpoint.npz", 3, 2, loss_mode="plain")


if __name__ == "__main__":
    unittest.main()
