import tempfile
import unittest

import jax
import numpy as np

from src import model, multisheet
from src.world import sequence
from train import train


class MultiSheetTests(unittest.TestCase):
    def test_parameter_count_matches_single_sheet(self):
        key = jax.random.PRNGKey(4)
        recurrent_count = sum(v.size for v in jax.tree.leaves(model.init(key)))
        two_count = sum(v.size for v in jax.tree.leaves(multisheet.init(key)))
        self.assertEqual(recurrent_count, two_count)
        self.assertEqual(multisheet.FAST_CHANNELS + multisheet.SLOW_CHANNELS, 8)

    def test_prediction_is_causal_and_feedback_is_intervention(self):
        frames, actions, _ = sequence(8, steps=6)
        params = multisheet.init(jax.random.PRNGKey(5))
        prediction, fast, slow = multisheet.trace(params, frames, actions)
        self.assertEqual(prediction.shape, (6, 16, 16, 1))
        self.assertEqual(fast.shape[-1], 6)
        self.assertEqual(slow.shape[-1], 2)
        modified = frames.copy()
        modified[-1] = 1 - modified[-1]
        np.testing.assert_array_equal(prediction, multisheet.predict(params, modified, actions))
        disconnected = multisheet.predict(params, frames, actions, 0.0)
        self.assertGreater(float(np.max(np.abs(prediction - disconnected))), 1e-7)

    def test_multisheet_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            train(seed=4, steps=2, batch_size=2, architecture="multisheet", output=folder)
            train(seed=4, steps=3, batch_size=2, architecture="multisheet", output=folder, resume=True)


if __name__ == "__main__":
    unittest.main()
