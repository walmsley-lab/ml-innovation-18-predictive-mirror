import tempfile
import unittest

import jax
import numpy as np

from evaluate import evaluate
from src import feedforward, model
from src.world import sequence
from train import train


class FeedforwardTests(unittest.TestCase):
    def test_causal_history_only(self):
        frames, actions, _ = sequence(9, steps=12)
        p = feedforward.init(jax.random.PRNGKey(1))
        predicted = np.asarray(feedforward.predict(p, frames, actions))
        mutated = frames.copy()
        mutated[-1] = 1 - mutated[-1]
        np.testing.assert_array_equal(predicted, np.asarray(feedforward.predict(p, mutated, actions)))
        self.assertEqual(predicted.shape, (12, 16, 16, 1))

    def test_equal_history_and_parameter_counts_reportable(self):
        key = jax.random.PRNGKey(4)
        a = sum(x.size for x in jax.tree.leaves(model.init(key)))
        b = sum(x.size for x in jax.tree.leaves(feedforward.init(key)))
        self.assertLess(abs(a - b) / a, 0.15)
        self.assertEqual(feedforward.HISTORY, 12)

    def test_training_and_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            train(seed=3, steps=2, batch_size=2, architecture="feedforward", output=folder)
        result = evaluate(feedforward.init(jax.random.PRNGKey(3)), range(3000, 3003), architecture="feedforward")
        self.assertIn("reappearance_location_mae_px", result["moving_object_moving_camera"]["model"])


if __name__ == "__main__":
    unittest.main()
