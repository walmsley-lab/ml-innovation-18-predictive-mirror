import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src import compact_predictive_mirror as model
from src import predictive_rgb_nca as nca


class CompactMirrorTests(unittest.TestCase):
    def setUp(self):
        self.params = model.init(jax.random.PRNGKey(11), channels=4)
        self.context = jax.random.uniform(jax.random.PRNGKey(12), (1, 3, 8, 12, 3))

    def test_compact_hidden_state_keeps_full_resolution_image(self):
        hidden, image = model.assimilate(self.params, self.context, factor=2)
        self.assertEqual(hidden.shape, (1, 4, 6, 4))
        np.testing.assert_array_equal(image, self.context[:, -1])
        full, _ = model.assimilate(self.params, self.context, factor=1)
        self.assertEqual(full.size, 4 * hidden.size)

    def test_first_lead_and_streaming_dream_match_rollout(self):
        state = model.assimilate(self.params, self.context)
        frames = []
        for _ in range(3):
            state, frame = model.dream(self.params, state)
            frames.append(frame)
        actual = model.rollout(self.params, self.context, 3)
        np.testing.assert_allclose(actual, jnp.stack(frames, axis=1), atol=1e-7)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0.)
        self.assertLessEqual(float(actual.max()), 1.)

    def test_correction_uses_committed_prediction_and_can_be_disabled(self):
        hidden, prior = model.initial_state(self.params, self.context[:, 0])
        prior = prior + .3
        observed = self.context[:, 1]
        expected, _ = nca.observe(self.params,
                                 (hidden, model.pool(prior)), model.pool(observed))
        corrected, current = model.observe(self.params, (hidden, prior), observed)
        np.testing.assert_allclose(corrected, expected, atol=1e-7)
        np.testing.assert_array_equal(current, observed)
        quiet, _ = nca.observe(self.params, (hidden, model.pool(observed)),
                               model.pool(observed))
        disabled, _ = model.observe(self.params, (hidden, prior), observed, feedback=False)
        np.testing.assert_allclose(disabled, quiet, atol=1e-7)
        self.assertGreater(float(jnp.max(jnp.abs(corrected - disabled))), 1e-7)

    def test_prior_history_affects_forecast_with_identical_last_frame(self):
        earlier = self.context.at[:, :-1].set(0.)
        params = {**self.params, "rgb_readout": jnp.full((4, 3), .5)}
        left = model.rollout(params, self.context, 2)
        right = model.rollout(params, earlier, 2)
        np.testing.assert_array_equal(earlier[:, -1], self.context[:, -1])
        self.assertGreater(float(jnp.max(jnp.abs(left - right))), 1e-6)

    def test_transport_preserves_detail_and_does_not_wrap(self):
        frame = jnp.zeros((1, 8, 12, 3)).at[:, 0, 3].set(1.)
        state = (model.initial_state(self.params, frame)[0], frame)
        center = jnp.full((9,), -40.).at[0].set(40.)
        params = {**self.params, "router": jnp.zeros((4, 9)), "router_bias": center}
        _, unchanged = model.dream(params, state)
        np.testing.assert_allclose(unchanged, frame, atol=1e-7)
        params["router_bias"] = jnp.full((9,), -40.).at[1].set(40.)
        _, shifted_up = model.dream(params, state)
        np.testing.assert_allclose(shifted_up, 0., atol=1e-7)

    def test_gradients_reach_memory_router_and_appearance(self):
        target = self.context[:, -1:] * .7
        loss = lambda p: jnp.mean((model.rollout(p, self.context, 2) - target)**2)
        value, grad = jax.value_and_grad(loss)(self.params)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(all(np.isfinite(g).all() for g in jax.tree.leaves(grad)))
        for key in ("perception", "router", "rgb_readout"):
            self.assertGreater(float(jnp.linalg.norm(grad[key])), 0.)

    def test_invalid_geometry_and_empty_context_fail(self):
        for factor in (0, -1, 2.5, 5):
            with self.assertRaises(ValueError):
                model.rollout(self.params, self.context, 2, factor=factor)
        with self.assertRaises(ValueError):
            model.rollout(self.params, self.context[:, :0], 2)
        with self.assertRaises(ValueError):
            model.rollout(self.params, self.context, 0)


if __name__ == "__main__":
    unittest.main()
