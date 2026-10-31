import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import cv2

from experiments.compare_horizons import (
    ARMS, add_score, draw_grid, forecast_all, load_models,
    new_scores, predictors, score_delayed, summaries, validation_positions)
from src.video_residual import init


class CompareHorizonTests(unittest.TestCase):
    def test_heldout_replay_never_uses_train_frames(self):
        indices = validation_positions(120, 8, 10, 1, "heldout")
        self.assertGreaterEqual(indices.start - 7, int(120 * .8))
        self.assertLess(indices.stop + 9, 120 + 1)
        self.assertRaises(ValueError, validation_positions, 60, 8, 50)

    def test_identical_context_and_exact_persistence_at_initialization(self):
        with tempfile.TemporaryDirectory() as root:
            from jax import tree
            params = init(jax.random.PRNGKey(7))
            for arm in ARMS:
                np.savez_compressed(Path(root) / f"{arm}.npz",
                                    **tree.map(np.asarray, params))
            models = load_models(root)
            forecasters = predictors(models, 5, 0.25)
            observed = np.random.default_rng(9).random((8, 12, 12, 1),
                                                       dtype=np.float32)
            result, ms = forecast_all(forecasters, observed, 5)
            self.assertGreaterEqual(ms, 0)
            for arm in ARMS:
                np.testing.assert_allclose(
                    result[arm], np.broadcast_to(observed[-1], (5, 12, 12, 1)),
                    atol=1e-7)
            altered_future = np.ones((5, 12, 12, 1), np.float32)
            self.assertFalse(np.allclose(result["plain"], altered_future))

    def test_delayed_live_scoring_only_when_future_frame_arrives(self):
        totals = new_scores((1, 3))
        last = np.zeros((4, 4, 1), np.float32)
        predictions = {arm: np.zeros((3, 4, 4, 1), np.float32)
                       for arm in ARMS}
        pending = [{"index": 10, "last": last, "forecast": predictions}]
        pending = score_delayed(pending, 10, last, totals, (1, 3))
        self.assertEqual(summaries(totals)["plain"]["1"]["scored_forecasts"], 0)
        pending = score_delayed(
            pending, 11, np.ones_like(last), totals, (1, 3))
        self.assertEqual(summaries(totals)["plain"]["1"]["scored_forecasts"], 1)
        self.assertEqual(summaries(totals)["plain"]["3"]["scored_forecasts"], 0)
        pending = score_delayed(
            pending, 13, np.ones_like(last), totals, (1, 3))
        self.assertEqual(summaries(totals)["plain"]["3"]["scored_forecasts"], 1)
        self.assertEqual(len(pending), 0)

    def test_render_is_synchronized_and_headless(self):
        current = np.full((12, 12, 1), .4, np.float32)
        predictions = {arm: np.repeat(current[None], 3, axis=0)
                       for arm in ARMS}
        canvas = draw_grid(current, predictions, (1, 3), cv2, panel_size=80,
                           actuals={1: current, 3: current}, fps=10)
        self.assertEqual(canvas.shape, (34 + 3*80, 75 + 5*80, 3))


if __name__ == "__main__":
    unittest.main()
