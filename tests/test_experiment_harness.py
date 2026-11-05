import json
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from train import adamw
from src import experiment_harness as harness


class ExperimentHarnessTests(unittest.TestCase):
    def test_leads_idiom(self):
        self.assertEqual(harness.leads(3), (1, 3))
        self.assertEqual(harness.leads(10), (1, 3, 5, 10))
        self.assertEqual(harness.leads(7), (1, 3, 5, 7))
        self.assertEqual(harness.leads(1), (1,))

    def test_init_moments_match_tree(self):
        params = {"w": jnp.ones((2, 3)), "b": jnp.ones((3,))}
        first, second = harness.init_moments(params)
        for store in (first, second):
            self.assertEqual(set(store), {"w", "b"})
            self.assertTrue(bool(jnp.all(store["w"] == 0)))
            self.assertEqual(store["w"].shape, (2, 3))

    def test_clipped_adamw_update_matches_inline_pattern(self):
        # A tiny quadratic whose gradient norm exceeds 1, so clipping engages.
        params = {"x": jnp.array([3.0, -4.0])}  # ||grad|| = 2*5 = 10 > 1
        batch = (jnp.array([1.0, 1.0]),)

        def loss(p, weights):
            return jnp.sum(weights * p["x"] ** 2)

        first, second = harness.init_moments(params)
        update = harness.clipped_adamw_update(loss)
        got_p, got_m, got_v, got_loss = update(params, first, second, batch, 1)

        # Recompute the exact inline step the runners wrote by hand.
        value, grad = jax.value_and_grad(lambda q: loss(q, *batch))(params)
        norm = jnp.sqrt(sum((g * g).sum() for g in jax.tree.leaves(grad)))
        grad = jax.tree.map(lambda g: g / jnp.maximum(norm, 1.0), grad)
        exp_p, exp_m, exp_v = adamw(params, grad, first, second, 1)

        np.testing.assert_allclose(np.asarray(got_p["x"]), np.asarray(exp_p["x"]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got_m["x"]), np.asarray(exp_m["x"]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got_v["x"]), np.asarray(exp_v["x"]), rtol=1e-6)
        np.testing.assert_allclose(float(got_loss), float(value), rtol=1e-6)
        # Clipping actually changed the raw gradient (norm was 10).
        self.assertGreater(float(norm), 1.0)

    def test_checkpoint_roundtrip_and_hash(self):
        params = {"w": jnp.arange(6.0).reshape(2, 3), "b": jnp.zeros((3,))}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.npz"
            np.savez(path, **{k: np.asarray(v) for k, v in params.items()})
            loaded, digest = harness.load_checkpoint(path)
            self.assertEqual(set(loaded), {"w", "b"})
            np.testing.assert_array_equal(np.asarray(loaded["w"]), np.asarray(params["w"]))
            self.assertEqual(len(digest), 64)
            # Hash is stable for identical bytes.
            self.assertEqual(digest, harness.load_checkpoint(path)[1])

    def test_write_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nested" / "metrics.json"
            written = harness.write_metrics(str(path), {"a": 1, "b": [2, 3]})
            self.assertEqual(written, path)
            text = path.read_text()
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(json.loads(text), {"a": 1, "b": [2, 3]})
            self.assertIsNone(harness.write_metrics(None, {"x": 1}))


if __name__ == "__main__":
    unittest.main()
