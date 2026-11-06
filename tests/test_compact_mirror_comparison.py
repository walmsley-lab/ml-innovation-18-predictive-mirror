import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from experiments import compact_mirror_comparison as study


class CompactComparisonTests(unittest.TestCase):
    def test_disjoint_splits_and_scene_construction(self):
        splits = study.split_ids(2, 2, 2, 2)
        self.assertFalse(set(splits["train"]) & set(splits["validation"]))
        self.assertFalse(set(splits["test"]) & set(splits["validation"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        frames, records, seeds = study.scenes(splits["test"])
        self.assertEqual(frames.shape, (6, 18, 32, 32, 3))
        self.assertEqual(len(seeds), 2)
        for i in (0, 3):
            np.testing.assert_array_equal(frames[i+1, 7], frames[i+2, 7])
            self.assertFalse(np.array_equal(frames[i+1, :7], frames[i+2, :7]))
            self.assertFalse(np.array_equal(frames[i+1, 8:], frames[i+2, 8:]))
        for seed in seeds:
            episode = study.world.episode(seed, 18, 32, world_margin=0)
            self.assertTrue(np.any(np.diff(episode["velocities"], axis=0)[7:] != 0))
        oracle = study.score(frames[:, 8:], frames, records, 8)
        self.assertEqual(oracle["rgb_mse"], 0.)
        self.assertEqual(oracle["occlusion"]["success_rate"], 1.)
        self.assertEqual(len(oracle["scene_mse"]), 2)
        other, _, _ = study.scenes(splits["validation"])
        self.assertFalse(any(np.array_equal(a, b) for a in frames for b in other))

    def test_step_zero_selection_and_future_targets_not_passed_to_predictor(self):
        validation, _, _ = study.scenes([10], horizon=6)
        calls = []

        def predict(params, prefix):
            calls.append(prefix.shape)
            self.assertEqual(prefix.shape[1], 8)
            return jnp.full((3, 6, 32, 32, 3), params["level"])

        def update(params, first, second, batch, step):
            return {"level": jnp.array(1.)}, first, second, jnp.array(1.)

        best, selected = study.fit({"level": jnp.array(.05)}, predict, update,
                                  [11], validation, 1, 1, 8, 6, 32)
        self.assertEqual(selected["selected_step"], 0)
        self.assertAlmostEqual(float(best["level"]), .05)
        self.assertEqual(len(calls), 2)

    def test_timing_uses_actual_batch_one_inputs(self):
        prefix = np.stack([np.full((8, 4, 4, 3), value, np.float32)
                           for value in (.2, .8)])
        seen = []

        def predict(x):
            self.assertEqual(x.shape, (1, 8, 4, 4, 3))
            seen.append(float(x.mean()))
            return np.repeat(x[:, -1:], 6, axis=1)

        predicted, timing = study.timed_forecasts(predict, prefix, 3)
        np.testing.assert_allclose(predicted[:, 0], prefix[:, -1])
        self.assertEqual(timing["timed_clips"], 2)
        self.assertEqual(len(timing["samples_ms"]), 6)
        self.assertLess(min(seen), .3)
        self.assertGreater(max(seen), .7)

    def test_gallery_pairs_exact_forecast_lead_and_error(self):
        frames, _, _ = study.scenes([10], horizon=6)
        prediction = frames[:, 8:].copy() * .8
        saved = []

        def capture(images, path, scale):
            saved.append(images)
            Path(path).write_bytes(b"test-image")

        with tempfile.TemporaryDirectory() as folder:
            with patch("experiments.verified_replay_gallery.save_panel", capture):
                entries = study.gallery(Path(folder), "model", prediction, frames, 8)
        for images, entry in zip(saved, entries):
            episode, lead = entry["episode"], entry["lead"]
            self.assertEqual(entry["target_frame"], 8 + lead - 1)
            np.testing.assert_array_equal(images[1], prediction[episode, lead-1])
            np.testing.assert_array_equal(images[2], frames[episode, 8+lead-1])
            np.testing.assert_allclose(images[3], np.abs(images[1] - images[2]))

    def test_end_to_end_freezes_artifacts_and_scores_all_references(self):
        with tempfile.TemporaryDirectory() as folder:
            result = study.run(folder, seeds=(1,), models=("compact",), steps=1,
                               validation_scenes=1, test_scenes=1, width=2,
                               horizon=6, ticks=1, repeats=3)
            path = Path(folder)
            self.assertEqual(result["protocol_sha256"], study.sha(path / "protocol.json"))
            self.assertEqual(result["selection_sha256"], study.sha(path / "selection.json"))
            self.assertEqual(set(result["results"]),
                             {*study.CONTROLS, "compact-seed-1"})
            protocol = json.loads((path / "protocol.json").read_text())
            self.assertEqual(protocol["model_seeds"], [1])
            for row in result["results"].values():
                artifact = row["predictions"]
                self.assertEqual(artifact["sha256"], study.sha(path / artifact["file"]))
                self.assertEqual(row["cost"]["timed_clips"], 3)
                self.assertIsNone(row["cost"]["peak_memory_mib"])
                with np.load(path / artifact["file"], allow_pickle=False) as saved:
                    self.assertEqual(saved["predictions"].shape, (3, 6, 32, 32, 3))
            row = result["results"]["compact-seed-1"]
            self.assertEqual(row["cost"]["state_payload_bytes"], (16*16*2+32*32*3)*4)
            self.assertEqual(set(row["diagnostics"]) - {"interpretation"},
                             {"history_reset", "error_feedback_disabled"})
            self.assertTrue((path / "index.html").exists())
            with self.assertRaisesRegex(ValueError, "empty"):
                study.run(folder, steps=1)

    def test_bad_budgets_and_forecasts_fail(self):
        with self.assertRaises(ValueError):
            study.split_ids(1_000_000, 1, 1, 1)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                study.run(folder, seeds=(1, 1))
        frames, records, _ = study.scenes([10])
        for value in (np.nan, np.inf, -1., 1.1):
            with self.assertRaises(ValueError):
                study.score(np.full_like(frames[:, 8:], value), frames, records, 8)


if __name__ == "__main__":
    unittest.main()
