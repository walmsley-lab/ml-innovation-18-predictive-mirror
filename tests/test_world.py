import unittest

import jax
import numpy as np

from src.model import init, predict
from src.world import batch, sequence


class TestPilot(unittest.TestCase):
    def test_deterministic_sequences_and_actions(self):
        x, actions, truth = sequence(21, 9)
        y, actions2, truth2 = sequence(21, 9)
        self.assertEqual(x.shape, (10, 16, 16, 1))
        self.assertEqual(actions.shape, (9, 1))
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(actions, actions2)
        np.testing.assert_array_equal(truth["world_x"], truth2["world_x"])
        self.assertTrue(np.all((actions >= -1) & (actions <= 1)))
        self.assertEqual(truth["visible_mask"].shape, x.shape)
        np.testing.assert_array_equal(truth["visible_mask"], x == 1.0)

    def test_training_loss_is_plain_observation_mse(self):
        from src.model import loss

        frames, actions = batch([11, 12], steps=5)
        p = init(jax.random.PRNGKey(1))
        forecasts = np.stack([
            np.asarray(predict(p, frames[i], actions[i])) for i in range(2)
        ])
        expected = np.mean((forecasts - frames[:, 1:]) ** 2)
        self.assertAlmostEqual(float(loss(p, frames, actions)), float(expected), places=6)

    def test_prediction_is_causal_and_finite(self):
        x, actions = batch([101, 102], steps=6)
        p = init(jax.random.PRNGKey(0))
        y = np.asarray(predict(p, x[0], actions[0]))
        self.assertEqual(y.shape, (6, 16, 16, 1))
        self.assertTrue(np.isfinite(y).all())
        shifted = x[0].copy()
        shifted[-1] = 1 - shifted[-1]
        np.testing.assert_array_equal(y, np.asarray(predict(p, shifted, actions[0])))


if __name__ == "__main__":
    unittest.main()
