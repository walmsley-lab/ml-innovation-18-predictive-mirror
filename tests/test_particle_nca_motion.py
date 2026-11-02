"""Causal particle NCA tests with a paired CPU smoke."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import particle_nca as model
from experiments.particle_nca_motion import controls, dataset, run


class ParticleTests(unittest.TestCase):
    def test_world_is_seeded(self):
        a = dataset((11, 12), 4, 5, 20, 2)
        np.testing.assert_array_equal(a, dataset((11, 12), 4, 5, 20, 2))
        self.assertEqual(a.shape, (2, 9, 2, 2))

    def test_distance_changes_messages(self):
        h = jnp.asarray([[[1., 0.], [0., 2.]]])
        near = jnp.asarray([[[.3, .3], [.31, .31]]])
        far = jnp.asarray([[[.05, .05], [.95, .95]]])
        self.assertGreater(
            float(np.abs(model.neighbor_messages(h, near)).sum()),
            float(np.abs(model.neighbor_messages(h, far)).sum()))

    def test_initial_persistence_and_no_future_input(self):
        x = dataset((11,), 4, 5, 20, 2)
        params = model.init(jax.random.PRNGKey(2), 4)
        forecast = np.asarray(model.rollout(params, jnp.asarray(x[:, :4]), 5))
        np.testing.assert_allclose(
            forecast, controls(x[:, :4], 5)["persistence"], atol=1e-6)
        changed = x.copy()
        changed[:, 4:] = 1.
        np.testing.assert_array_equal(forecast, np.asarray(
            model.rollout(params, jnp.asarray(changed[:, :4]), 5)))

    def test_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(folder, seeds=(3,), steps=1, batch_size=1,
                         heldout=2, context=4, horizon=3, size=20,
                         particles=2, channels=4)
            self.assertEqual(set(result["runs"][0]["metrics"]),
                {"messages", "no_messages", "persistence", "constant_velocity"})
            self.assertTrue((Path(folder)/"seed-3-messages.npz").exists())


class DisjointSplitTests(unittest.TestCase):
    def test_default_seeds_are_disjoint(self):
        from experiments.particle_nca_motion import check_disjoint
        check_disjoint((11, 23, 37), 100, 4, 16)

    def test_overlapping_seed_is_refused(self):
        from experiments.particle_nca_motion import check_disjoint
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_disjoint((0, 599), 1, 4, 16)


if __name__ == "__main__":
    unittest.main()
