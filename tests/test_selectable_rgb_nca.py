import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from src import selectable_rgb_nca as nca
from experiments.nca_selector_ablation import experiment


class SelectorTests(unittest.TestCase):
    def test_local_and_relative_offsets(self):
        x = jnp.zeros((1, 7, 7, 1), jnp.float32).at[0, 3, 3, 0].set(9.0)
        local = np.asarray(nca.select(x, "local"))
        self.assertAlmostEqual(float(local[0, 3, 3, 0]), 1.0)
        self.assertAlmostEqual(float(local[0, 4, 4, 0]), 1.0)
        relative = np.asarray(nca.select(x, "relative", offsets=((0, -2),)))
        self.assertEqual(float(relative[0, 3, 5, 0]), 9.0)
        self.assertEqual(float(relative[0, 3, 3, 0]), 0.0)
        self.assertEqual(float(relative[0, 3, 0, 0]), 0.0)

    def test_reference_selection_is_nonlocal_and_explicit(self):
        x = jnp.arange(25, dtype=jnp.float32).reshape(1, 5, 5, 1)
        refs = nca.reference_map(5, 5)
        values = np.asarray(nca.select(x, "reference", references=refs))
        self.assertEqual(values[0, 0, 0, 0], 24)
        self.assertEqual(values[0, 4, 4, 0], 0)
        with self.assertRaises(ValueError):
            nca.select(x, "reference")

    def test_causal_and_identical_persistence_initialization(self):
        params = nca.init(jax.random.PRNGKey(3), channels=8)
        context = jnp.asarray(np.random.default_rng(5).random(
            (1, 4, 16, 16, 3), dtype=np.float32))
        refs = nca.reference_map(16, 16)
        for mode in nca.MODES:
            output = np.asarray(nca.rollout(params, context, 3, 2,
                            mode, refs if mode == "reference" else None))
            np.testing.assert_allclose(output,
                np.broadcast_to(np.asarray(context[:, -1:]), output.shape), atol=1e-7)

    def test_gradients_relative_reference(self):
        params = nca.init(jax.random.PRNGKey(1), channels=8)
        frames = jnp.asarray(np.random.default_rng(3).random(
            (1, 7, 16, 16, 3), dtype=np.float32))
        for mode in ("relative", "reference"):
            refs = nca.reference_map(16, 16) if mode == "reference" else None
            grad = jax.grad(lambda p: nca.loss(p, frames, 4, 3, 1, mode, refs))(params)
            self.assertGreater(float(jnp.abs(grad["readout"]).sum()), 0)

    def test_three_arm_training_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "rgb.npz"
            np.savez_compressed(file, frames=np.random.default_rng(1).random(
                (120, 16, 16, 3), dtype=np.float32))
            report = experiment(file, Path(directory)/"out", steps=1,
                                batch_size=1, context=4, horizon=3, ticks=1)
            self.assertEqual(set(report["arms"]), set(nca.MODES))
            self.assertTrue((Path(directory)/"out"/"reference.npz").exists())


if __name__ == "__main__":
    unittest.main()
