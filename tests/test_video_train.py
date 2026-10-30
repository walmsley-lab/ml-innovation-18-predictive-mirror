import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from src import video_residual
from video_train import windows, fit
from mirror import predict_next, initial_state


class VideoTrainingTests(unittest.TestCase):
    def test_initial_prediction_is_persistence_and_causal(self):
        p = video_residual.init(jax.random.PRNGKey(3))
        frames = np.linspace(0, 1, 10*16*16).reshape(10, 16, 16, 1).astype(np.float32)
        action = jnp.zeros((9, 1), jnp.float32)
        y = np.asarray(video_residual.predict(p, frames, action))
        np.testing.assert_allclose(y, frames[:-1], rtol=0, atol=1e-7)
        modified = frames.copy()
        modified[-1] = 0
        np.testing.assert_array_equal(y, np.asarray(video_residual.predict(p, modified, action)))

    def test_temporal_holdout_and_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.random.default_rng(15).random((120, 16, 16, 1), dtype=np.float32)
            train_idx, val_idx = windows(raw, 8)
            self.assertGreater(int(val_idx[0])-int(train_idx[-1])-8, 8)
            inp = Path(tmp) / "data.npz"
            np.savez_compressed(inp, frames=raw)
            metrics = fit(inp, Path(tmp) / "out", steps=2, batch_size=2)
            self.assertEqual(metrics["architecture"], "video_residual")
            self.assertTrue((Path(tmp) / "out" / "parameters.npz").is_file())

    def test_live_adapter_is_compatible(self):
        params = video_residual.init(jax.random.PRNGKey(4))
        state = initial_state("video_residual", 32)
        frame = np.zeros((32, 32, 1), np.float32)
        _, predicted = predict_next(params, "video_residual", state, frame)
        np.testing.assert_allclose(predicted, frame)


if __name__ == "__main__":
    unittest.main()
