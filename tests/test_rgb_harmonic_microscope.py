"""Spatial spectral basis and frozen-RGB microscope smoke tests."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca
from src import rgb_harmonics as hm
from experiments.rgb_harmonic_microscope import run


class HarmonicsTests(unittest.TestCase):
    def test_path_grid_modes_are_orthonormal_eigenvectors(self):
        eig, modes = hm.lattice_modes(4, 5, 6)
        flat = modes.reshape(len(eig), -1)
        np.testing.assert_allclose(flat @ flat.T, np.eye(len(eig)), atol=1e-12)
        assert eig.min() > 0
        for value, mode in zip(eig, modes):
            v = mode
            # Open boundary 4-neighbor combinatorial Laplacian.
            lap = np.zeros_like(v)
            lap[:-1] += v[:-1] - v[1:]
            lap[1:] += v[1:] - v[:-1]
            lap[:, :-1] += v[:, :-1] - v[:, 1:]
            lap[:, 1:] += v[:, 1:] - v[:, :-1]
            np.testing.assert_allclose(lap, value * v, atol=1e-10)

    def test_projection_energy_and_shuffle(self):
        eig, modes = hm.lattice_modes(4, 4, 5)
        frames = np.broadcast_to(modes[0][None, None, :, :, None], (2, 3, 4, 4, 1)).copy()
        s = hm.spectrum(frames, modes)
        self.assertAlmostEqual(s["selected_fraction"], 1, places=6)
        self.assertGreater(s["fractions"][0], .999)
        shuffled = hm.shuffle_sites(frames, 13)
        self.assertAlmostEqual(float(np.square(shuffled).sum()),
                               float(np.square(frames).sum()), places=6)

    def test_trace_matches_existing_forecast_and_no_future_peek(self):
        frames = np.random.default_rng(3).random((1, 7, 8, 8, 3),
                                                  dtype=np.float32)
        params = nca.init(jax.random.PRNGKey(1), 4)
        seen, dreamed, prediction = hm.trace(
            params, jnp.asarray(frames[:, :4]), 3, 1)
        np.testing.assert_allclose(
            np.asarray(prediction),
            np.asarray(nca.rollout(params, jnp.asarray(frames[:, :4]), 3, 1)),
            atol=1e-6)
        self.assertEqual(seen.shape[1], 4)
        self.assertEqual(dreamed.shape[1], 3)
        changed = frames.copy()
        changed[:, 4:] = 0
        np.testing.assert_array_equal(
            np.asarray(prediction), np.asarray(
                hm.trace(params, jnp.asarray(changed[:, :4]), 3, 1)[2]))

    def test_cli_runner_step_zero_smoke(self):
        params = nca.init(jax.random.PRNGKey(2), 4)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "weights.npz"
            np.savez_compressed(checkpoint, **jax.tree.map(np.asarray, params))
            report = run(checkpoint, Path(folder) / "out",
                         episodes=1, context=4, horizon=2, size=16,
                         ticks=1, modes=4)
            self.assertEqual(set(report["arms"]),
                             {"checkpoint", "random_initialization",
                              "checkpoint_shuffled"})
            self.assertTrue((Path(folder) / "out" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
