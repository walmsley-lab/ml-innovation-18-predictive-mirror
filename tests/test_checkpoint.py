import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np

from src.checkpoint import load, save
from src.model import init
from train import train


class CheckpointTests(unittest.TestCase):
    def test_round_trip_complete_optimizer_state(self):
        params = init(jax.random.PRNGKey(4))
        zeros = jax.tree.map(lambda value: value * 0, params)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.npz"
            save(path, params, zeros, zeros, 7, 42, 2)
            recovered = load(path, 42, 2)
            self.assertEqual(recovered[-1], 7)
            for name in params:
                np.testing.assert_array_equal(recovered[0][name], params[name])
            with self.assertRaises(ValueError):
                load(path, 43, 2)

    def test_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as folder:
            whole = Path(folder) / "whole"
            resumed = Path(folder) / "resumed"
            train(seed=4, steps=4, batch_size=2, output=str(whole), checkpoint_every=2)
            train(seed=4, steps=2, batch_size=2, output=str(resumed), checkpoint_every=1)
            train(seed=4, steps=4, batch_size=2, output=str(resumed), checkpoint_every=1, resume=True)
            with np.load(whole / "parameters.npz") as a, np.load(resumed / "parameters.npz") as b:
                for key in a.files:
                    np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
