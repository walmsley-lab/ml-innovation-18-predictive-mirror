"""Causal contracts and end-to-end smoke for embodied arithmetic and touch."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import embodied_arithmetic_world as world
from src import embodied_arithmetic_model as model
from experiments import embodied_causal_curriculum as campaign


class EmbodiedCausalTests(unittest.TestCase):
    def test_recursive_arithmetic_and_invalid_tokens(self):
        self.assertEqual(world.interpret("Z"), 0)
        self.assertEqual(world.interpret("S(S(Z))"), 2)
        self.assertEqual(world.interpret("ADD(S(S(Z)),S(Z))"), 3)
        self.assertEqual(world.interpret("NEG(ADD(S(Z),S(S(Z))))"), -3)
        for text in ("S(", "ADD(S(Z))", "X(Z)", "S(Z))"):
            with self.assertRaises(ValueError):
                world.parse(text)

    def test_sensor_alignment_and_no_oracle_input(self):
        for kind in ("spring", "touch"):
            example = world.episode(17, kind, steps=8, size=8, blind=True)
            self.assertEqual(example["observed"].shape, (8, 196))
            self.assertEqual(example["target"].shape, (8, 196))
            self.assertEqual(example["tree"].shape, (8, 16, 3))
            self.assertEqual(example["shape"].shape, (8, 8, 8))
            self.assertEqual(example["motor_label"].shape, (8,))
            self.assertFalse("latent_state" in example)
            self.assertFalse("physics" in example)
            self.assertTrue(np.isfinite(example["target"]).all())

    def test_blind_mask_and_negative_control(self):
        data = world.dataset(101, 4, "touch", steps=8, size=8, blind="yes")
        untouched = campaign.prepare(data, np.arange(4),
                                    np.random.default_rng(4), "paired", False)
        shifted = campaign.prepare(data, np.arange(4),
                                  np.random.default_rng(4), "paired", False,
                                  "touch_time_shifted")
        self.assertFalse(np.any(untouched[4][..., 0]))
        np.testing.assert_equal(untouched[5], shifted[5])
        self.assertFalse(np.array_equal(untouched[0][..., -2:],
                                        shifted[0][..., -2:]))

    def test_learned_grammar_and_recurrent_shapes(self):
        data = world.dataset(200, 2, "touch", steps=6, size=8, blind="yes")
        args = campaign._tensor(campaign.prepare(
            data, np.arange(2), np.random.default_rng(0), "paired", False))
        params = model.init(jax.random.PRNGKey(3), 8, 8)
        predictions, contours, values = model.rollout(params, *args[:5], 8)
        self.assertEqual(predictions.shape, (2, 6, 196))
        self.assertEqual(contours.shape, (2, 6, 8, 8))
        self.assertEqual(values.shape, (2, 6))
        scalar, grad = jax.value_and_grad(model.loss)(
            params, *args, 8)
        self.assertTrue(np.isfinite(np.asarray(scalar)))
        self.assertTrue(all(np.isfinite(np.asarray(g)).all()
                            for g in jax.tree.leaves(grad)))

    def test_train_and_heldout_blind_touch(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = campaign.run(
                tmp, seeds=(11,), spring_steps=1, touch_steps=1,
                steps_per_episode=6, size=8, width=8, batch_size=2,
                train_count=4, val_count=2, test_count=2)
            self.assertEqual(report["arm_names"],
                             ["paired", "desynchronized_touch"])
            for arm in report["results"][0]["arms"].values():
                self.assertTrue(np.isfinite(arm["validation_loss"]))
                blind = arm["test"]["touch_blind"]["available"]
                self.assertEqual(blind["observed_vision_fraction"], 0.)
                self.assertTrue(np.isfinite(blind["unprobed_shape_mse"]))
                self.assertTrue(np.isfinite(blind["unprobed_shape_prior_mse"]))
            self.assertTrue((Path(tmp) / "metrics.json").exists())
            self.assertTrue((Path(tmp) / "seed-11-paired.npz").exists())


if __name__ == "__main__":
    unittest.main()
