import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from experiments.circus_curriculum import model, probes, world
from experiments.circus_curriculum import __main__ as study


class HypergeometryWorldTests(unittest.TestCase):
    def test_true_4d_topology_and_rotations(self):
        cube, edges = world.shape("tesseract", np.ones(4))
        self.assertEqual(cube.shape, (16, 4))
        self.assertEqual(edges.shape, (32, 2))
        cross, cross_edges = world.shape("16cell", np.ones(4))
        self.assertEqual(cross.shape, (8, 4))
        self.assertEqual(cross_edges.shape, (24, 2))
        r = world.rotation(4, (0, 3), .7)
        np.testing.assert_allclose(r @ r.T, np.eye(4), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.)
        moved = cube @ r.T
        np.testing.assert_allclose(np.linalg.norm(moved[:, None] - moved[None], axis=-1),
                                   np.linalg.norm(cube[:, None] - cube[None], axis=-1), atol=1e-12)
        self.assertFalse(np.allclose(moved[:, 3], cube[:, 3]))

    def test_projection_and_prefix_do_not_depend_on_future_duration(self):
        short = world.episode(11, steps=5, size=16)
        long = world.episode(11, steps=9, size=16)
        np.testing.assert_array_equal(short["frames"], long["frames"][:5])
        self.assertEqual(short["frames"].shape, (5, 16, 16, 3))
        self.assertTrue(np.isfinite(short["frames"]).all())
        self.assertGreaterEqual(short["frames"].min(), 0.)
        self.assertLessEqual(short["frames"].max(), 1.)
        self.assertFalse(np.array_equal(short["frames"][0], short["frames"][-1]))
        points = np.array([[1., 0., 0., 0.], [1., 0., 0., 1.]])
        projected = world.project(points, np.eye(3))
        self.assertGreater(projected[1, 0], projected[0, 0])

    def test_ood_changes_one_factor_and_combined_changes_all(self):
        records = {case: world.episode(19, 4, 16, case) for case in world.CASES}
        base = records["iid"]["truth"]
        for case, record in records.items():
            truth = record["truth"]
            np.testing.assert_array_equal(base["phases"], truth["phases"])
            np.testing.assert_array_equal(base["speeds"], truth["speeds"])
            self.assertEqual(truth["shape"] != base["shape"], case in ("object", "combined"))
            self.assertEqual(not np.array_equal(truth["planes"], base["planes"]),
                             case in ("rotation", "combined"))
            self.assertEqual(not np.array_equal(truth["camera"], base["camera"]),
                             case in ("viewpoint", "combined"))
            if case in ("viewpoint", "combined"):
                self.assertTrue(np.all(np.abs(truth["view_angles"]) >= .9))
            else:
                self.assertTrue(np.all(np.abs(truth["view_angles"]) <= .3))
        self.assertFalse(set(world.TRAIN_PLANES) & set(world.HELD_OUT_PLANES))


class CircusLatentTests(unittest.TestCase):
    def test_latent_dimension_alignment_and_open_loop_gradients(self):
        params = model.init(jax.random.PRNGKey(1), latent_dim=16)
        x = jnp.asarray(world.batch([2, 3], steps=5, size=16))
        state = model.encode_history(params, x[:, :3])
        self.assertEqual(state.shape, (2, 16))
        # A nonzero readout exercises the actual future transition and index.
        params = {**params, "readout": jnp.full((16, 8*8*3), .03)}
        latent, first = model.dream(params, state, x[:, 2])
        predicted = model.rollout(params, x[:, :3], 2)
        np.testing.assert_allclose(first, predicted[:, 0], atol=1e-7)
        changed_future = x.at[:, 3:].set(0.)
        np.testing.assert_array_equal(predicted, model.rollout(params, changed_future[:, :3], 2))
        value, gradient = jax.value_and_grad(
            lambda p: jnp.mean((model.rollout(p, x[:, :3], 2) - x[:, 3:])**2))(params)
        self.assertTrue(np.isfinite(float(value)))
        for name in ("encoder", "recurrent", "readout"):
            self.assertTrue(np.isfinite(gradient[name]).all())
            self.assertGreater(float(jnp.linalg.norm(gradient[name])), 0.)

    def test_history_is_used_and_bad_geometry_fails(self):
        params = model.init(jax.random.PRNGKey(3), latent_dim=16)
        x = jnp.asarray(world.batch([11], steps=4, size=16))
        reset = jnp.repeat(x[:, -1:], 4, axis=1)
        left, right = model.encode_history(params, x), model.encode_history(params, reset)
        self.assertGreater(float(jnp.max(jnp.abs(left-right))), 1e-7)
        with self.assertRaises(ValueError):
            model.rollout(params, jnp.ones((1, 3, 18, 18, 3)), 2)
        with self.assertRaises(ValueError):
            model.rollout(params, x[:, :1], 2)

    def test_probe_fit_is_frozen_and_constant_targets_have_undefined_r2(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 3))
        matrix = rng.normal(size=(3, 2))
        fitted = probes.fit(x, x @ matrix, penalty=.001)
        test = rng.normal(size=(10, 3))
        prediction = probes.predict(fitted, test)
        self.assertLess(probes.score(prediction, test @ matrix)["mse"], 1e-6)
        result = probes.score(np.ones((3, 2)), np.ones((3, 2)))
        self.assertEqual(result["r2_by_coordinate"], [None, None])
        with self.assertRaises(ValueError):
            probes.fit(x, np.full((40, 2), np.nan))


class CircusRunnerTests(unittest.TestCase):
    def test_curricula_preserve_total_clip_budget(self):
        ordinary = np.zeros((8, 5, 16, 16, 3), np.float32)
        geometry = np.ones_like(ordinary)
        for curriculum in study.CURRICULA:
            pool = study.training_pool(ordinary, geometry, curriculum)
            self.assertEqual(pool.shape, ordinary.shape)
        mixed = study.training_pool(ordinary, geometry, "mixed")
        self.assertEqual(float(mixed.mean()), .5)

    def test_end_to_end_keeps_test_and_probe_splits_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            report = study.run(folder, seeds=(1,), steps=1, batch_size=2,
                               train_episodes=4, validation_episodes=2, test_episodes=2,
                               probe_episodes=4, context=3, horizon=2, size=16, latent_dim=4)
            folder = Path(folder)
            protocol = json.loads((folder / "protocol.json").read_text())
            ids = list(protocol["split_episode_ids"].values())
            for i, first in enumerate(ids):
                for second in ids[i+1:]:
                    self.assertFalse(set(first) & set(second))
            schedule = np.asarray(protocol["training_schedule"])
            self.assertTrue(np.all(schedule[:, 0] < 2))
            self.assertTrue(np.all(schedule[:, 1] >= 2))
            self.assertEqual(report["protocol_sha256"], study.sha(folder / "protocol.json"))
            self.assertEqual(report["selection_sha256"], study.sha(folder / "selection.json"))
            self.assertEqual(len(report["results"]), 3)
            for row in report["results"].values():
                self.assertEqual(set(row["cases"]), {*world.CASES, "ordinary"})
                self.assertEqual(set(row["latent_probes"]["results"]),
                                 {"trained_latent", "random_latent", "last_image", "image_history"})
                self.assertIn(row["selection"]["selected_step"], (0, 1))
                for case in row["cases"].values():
                    item = case["predictions"]
                    self.assertEqual(item["sha256"], study.sha(folder / item["file"]))
            self.assertEqual(len(report["ordinary_transfer"]), 1)
            self.assertEqual(len(report["ordinary_transfer"][0]["paired_episode_differences"]), 2)
            with self.assertRaisesRegex(ValueError, "empty"):
                study.run(folder, steps=1)


if __name__ == "__main__":
    unittest.main()
