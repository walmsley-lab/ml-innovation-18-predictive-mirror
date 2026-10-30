import unittest

import jax
import jax.numpy as jnp
import numpy as np

from experiments.active_field import advance, compare, episode, gaze_action, observe, reset
from src import model, plastic_field


class ActiveFieldTests(unittest.TestCase):
    def test_zero_plasticity_matches_base_network_step(self):
        params = model.init(jax.random.PRNGKey(6))
        state = plastic_field.initial()
        frame, _ = observe(reset(3))
        action = jnp.asarray([[0]], jnp.float32)
        next_state, y = plastic_field.step(params, state, jnp.asarray(frame)[None], action, 0)
        hidden, expected = model.step(params, state[0], jnp.asarray(frame)[None], action)
        np.testing.assert_allclose(next_state[0], hidden, rtol=0, atol=0)
        np.testing.assert_allclose(y, expected, rtol=0, atol=0)
        self.assertEqual(float(jnp.sum(next_state[1])), 0.0)

    def test_local_gain_changes_without_updating_model_params(self):
        params = model.init(jax.random.PRNGKey(7))
        before = jax.tree.map(np.asarray, params)
        state = plastic_field.initial()
        frame, _ = observe(reset(9))
        for _ in range(4):
            state, _ = plastic_field.step(params, state, jnp.asarray(frame)[None],
                                         jnp.zeros((1, 1)), plasticity_rate=0.1)
        self.assertGreater(float(jnp.sum(jnp.abs(state[1]))), 0.0)
        for key in params:
            np.testing.assert_array_equal(before[key], params[key])

    def test_policy_has_no_truth_input_and_controls_have_equal_budget(self):
        params = model.init(jax.random.PRNGKey(3))
        frame, _ = observe(reset(8))
        self.assertIn(gaze_action(params, plastic_field.initial(), frame), (-1, 0, 1))
        report = compare(params, [9], steps=3)
        self.assertEqual(len(report), 6)
        self.assertTrue(all(v["observation_budget"] == 3 for v in report.values()))
        self.assertEqual(episode(params, 9, "random", steps=3),
                         episode(params, 9, "random", steps=3))
        with self.assertRaises(ValueError):
            advance(reset(9), 2)


if __name__ == "__main__":
    unittest.main()
