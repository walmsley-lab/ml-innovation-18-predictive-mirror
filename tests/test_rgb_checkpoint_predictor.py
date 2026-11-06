"""CPU-safe smoke checks that every supported cloud checkpoint can be viewed."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from src import rgb_checkpoint_predictor as predictor
from src import predictive_rgb_nca as nca
from src import rgb_grounded_nca as grounded
from src import tiny_cnn_video as cnn
from src import rgb_motion_transport as transport


class CheckpointPredictorTests(unittest.TestCase):
    def test_all_models_roll_out_future_rgb(self):
        params = {
            "nca": nca.init(jax.random.PRNGKey(1), 4),
            "nca_blend": grounded.init(jax.random.PRNGKey(2), 4, "blend"),
            "cnn": cnn.init(jax.random.PRNGKey(3), 4),
            "transport": transport.init(jax.random.PRNGKey(4), 4),
        }
        observations = jnp.ones((1, 3, 8, 8, 3), jnp.float32) * .2
        for model, weights in params.items():
            with self.subTest(model=model):
                output = jax.jit(lambda x: predictor.rollout(
                    weights, x, horizon=2, ticks=1, model=model))(observations)
                self.assertEqual(output.shape, (1, 2, 8, 8, 3))
                self.assertTrue(np.isfinite(np.asarray(output)).all())

    def test_wrong_checkpoint_fails_closed(self):
        standard = nca.init(jax.random.PRNGKey(1), 4)
        blend = grounded.init(jax.random.PRNGKey(2), 4, "blend")
        with self.assertRaisesRegex(ValueError, "missing"):
            predictor.validate(standard, "cnn")
        with self.assertRaisesRegex(ValueError, "requires"):
            predictor.validate(blend, "nca")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            predictor.validate(standard, "unknown")

    def test_no_future_input(self):
        weights = cnn.init(jax.random.PRNGKey(1), 3)
        history = jnp.ones((1, 2, 8, 8, 3), jnp.float32) * .3
        first = predictor.rollout(weights, history, 3, model="cnn")
        second = predictor.rollout(weights, history.at[:,-1].set(0), 3, model="cnn")
        self.assertEqual(first.shape, second.shape)
        self.assertFalse(np.allclose(np.asarray(first), np.asarray(second)))


if __name__ == "__main__":
    unittest.main()
