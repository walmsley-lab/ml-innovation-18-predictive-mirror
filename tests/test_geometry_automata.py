"""Local dynamics, causal sensing and geometry transfer contracts."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import geometry_automata as world
from src import geometry_perceiver as model
from experiments.geometry_automata import run


class GeometryAutomataTests(unittest.TestCase):
    def test_explicit_arbitrary_geometry_and_invalid_edges(self):
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
        edges = np.array([[0, 1], [1, 3], [3, 2]], np.int32)
        valid, paths = world.geometry(xy, edges)
        self.assertEqual(valid.shape, (4, 2))
        self.assertEqual(paths.shape, (3, 2))
        for invalid in (np.array([[0, 1], [1, 0], [2, 3]]),
                        np.array([[0, 1], [2, 3]])):
            with self.assertRaises(ValueError):
                world.geometry(xy, invalid)

    def test_permutation_equivariance_and_equal_opposite_internal_flux(self):
        coords, edges = world.generate(17, 13, "irregular")
        rng = np.random.default_rng(19)
        q = rng.normal(0, .1, coords.shape).astype(np.float32)
        v = rng.normal(0, .1, coords.shape).astype(np.float32)
        f = np.zeros_like(q)
        next_q, next_v = world.step(coords, edges, q, v, f,
                                     tether=0., damping=0.)
        np.testing.assert_allclose(next_v.sum(axis=0), v.sum(axis=0), atol=1e-6)
        permutation = rng.permutation(len(coords))
        inverse = np.argsort(permutation)
        transformed_edges = inverse[edges]
        new_q, new_v = world.step(coords[permutation], transformed_edges,
                                   q[permutation], v[permutation],
                                   f[permutation], tether=0., damping=0.)
        np.testing.assert_allclose(new_q, next_q[permutation], atol=1e-6)
        np.testing.assert_allclose(new_v, next_v[permutation], atol=1e-6)

    def test_externally_caused_contact_propagates(self):
        record = world.episode(71, steps=12, nodes=10, kind="ring")
        no_actions = world.episode(71, steps=12, nodes=10, kind="ring", action=False)
        self.assertTrue(np.any(np.abs(record["truth"][-1]) > 1e-6))
        np.testing.assert_allclose(no_actions["truth"], 0., atol=0.)
        self.assertGreater(np.abs(record["truth"][-1]).sum(), 0.)
        self.assertLess(record["sensor_mask"].sum(), np.prod(record["truth"].shape))

    def test_future_sensors_do_not_leak_into_forecast(self):
        sample = world.episode(82, steps=8, nodes=13)
        p = model.init(jax.random.PRNGKey(1), 8)
        obs = jnp.asarray(sample["observed"])
        mask = jnp.asarray(sample["sensor_mask"])
        force = jnp.asarray(sample["forces"])
        graph = jnp.asarray(sample["adjacency"])
        issued = np.asarray(model.predict(p, obs, mask, force, graph))
        altered = obs.at[1:].set(1000.)
        after = np.asarray(model.predict(p, altered, mask, force, graph))
        np.testing.assert_array_equal(issued[0], after[0])
        self.assertTrue(np.isfinite(np.asarray(model.loss(p, obs, mask,
                                                           force, graph))))

    def test_graph_model_permutation_equivariance(self):
        sample = world.episode(96, steps=8, nodes=13)
        p = model.init(jax.random.PRNGKey(4), 8)
        keys = ("observed", "sensor_mask", "forces", "adjacency")
        issued = np.asarray(model.predict(p, *(jnp.asarray(sample[k])
                                               for k in keys)))
        perm = np.random.default_rng(13).permutation(13)
        transformed = (sample["observed"][:, perm],
                       sample["sensor_mask"][:, perm],
                       sample["forces"][:, perm],
                       sample["adjacency"][np.ix_(perm, perm)])
        forecast = np.asarray(model.predict(p, *(jnp.asarray(k)
                                                 for k in transformed)))
        np.testing.assert_allclose(forecast, issued[:, perm], atol=1e-6)

    def test_smoke_training_and_causal_stress(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(folder, seeds=(11,), steps=2, channels=8)
            self.assertEqual(result["seeds"], [11])
            arms = result["reports"][0]["arms"]
            self.assertEqual(set(arms), {"messages", "no_messages"})
            for arm in arms.values():
                self.assertTrue(np.isfinite(arm["validation_loss"]))
                for domain in arm["test"].values():
                    self.assertTrue(np.isfinite(domain["normal"]["sensed_mse"]))
                    self.assertTrue(np.isfinite(domain["no_sensors"]["sensed_mse"]))
            self.assertTrue((Path(folder) / "metrics.json").exists())
            self.assertTrue((Path(folder) / "seed-11-messages.npz").exists())


if __name__ == "__main__":
    unittest.main()
