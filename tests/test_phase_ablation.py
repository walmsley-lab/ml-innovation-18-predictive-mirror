import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import oscillatory_video
from experiments.phase_ablation import run


class OscillatorTests(unittest.TestCase):
    def test_coupling_responds_to_phase_differences(self):
        phase = jnp.zeros((1, 8, 8, 1), jnp.float32)
        phase = phase.at[0, 4, 4, 0].set(1.0)
        coupling = np.asarray(oscillatory_video.neighbor_alignment(phase))
        self.assertNotEqual(float(coupling[0, 4, 5, 0]), 0.0)

    def test_initial_forecasts_are_persistence_and_causal(self):
        params = oscillatory_video.init(jax.random.PRNGKey(5))
        frames = np.random.default_rng(8).random((7, 16, 16, 1), dtype=np.float32)
        action = jnp.zeros((6, 1), jnp.float32)
        p = np.asarray(oscillatory_video.predict(params, jnp.asarray(frames), action))
        np.testing.assert_allclose(p, frames[:-1], atol=1e-7)
        altered = frames.copy()
        altered[-1] = 0
        np.testing.assert_array_equal(
            p, np.asarray(oscillatory_video.predict(params, jnp.asarray(altered), action)))

    def test_cpu_smoke_all_three_matched_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "frames.npz"
            np.savez_compressed(recording, frames=np.random.default_rng(5).random(
                (120, 16, 16, 1), dtype=np.float32))
            result = run(recording, Path(directory) / "out", steps=2, batch_size=2)
            self.assertEqual(set(result["arms"]),
                             {"baseline", "oscillator_uncoupled", "oscillator_coupled"})
            self.assertTrue((Path(directory) / "out" / "oscillator_coupled.npz").exists())


if __name__ == "__main__":
    unittest.main()
