import unittest
import numpy as np
from src import neuromatch_beliefs as m


class NeuromatchBeliefTests(unittest.TestCase):
    def test_kalman_missing_observations(self):
        observations = [1., 2., 3., None, None, 6.]
        filtered, covariance = m.kalman_track(observations, process_variance=0.001)
        self.assertEqual(filtered.shape, (6, 2))
        self.assertTrue(np.isfinite(covariance).all())
        self.assertAlmostEqual(filtered[3, 0], 4., delta=.3)
        self.assertAlmostEqual(m.hold_last(observations)[3], 3.)
        with self.assertRaises(ValueError):
            m.kalman_track([1, float("nan")])

    def test_memory_tracks_appearance_through_crossing_and_occlusion(self):
        slots = m.update_object_memory(None, [(1, .1), (9, .9)], 0)
        for t in range(1, 12):
            observations = [(1+t, .1), (9-t, .9)]
            if t in (4, 5):
                observations = observations[1:]
            if t % 2:
                observations = observations[::-1]
            slots = m.update_object_memory(slots, observations, t)
        self.assertAlmostEqual(slots[0]["position"], 12., delta=1.1)
        self.assertAlmostEqual(slots[1]["position"], -2., delta=1.1)

    def test_active_view_expected_gain(self):
        strong = np.array([[.99, .01], [.01, .99]])
        weak = np.array([[.55, .45], [.45, .55]])
        choice, gains = m.select_active_view([.5, .5], {"weak": weak, "strong": strong})
        self.assertEqual(choice, "strong")
        self.assertGreater(gains["strong"], gains["weak"])
        self.assertAlmostEqual(m.expected_information_gain([1., 0.], strong), 0.)

    def test_replay_retention_uses_existing_metrics_and_pairs(self):
        stages = [{"task": "A", "heldout_test": {"A": .1}},
                  {"task": "B", "heldout_test": {"A": .3}},
                  {"task": "A", "heldout_test": {"A": .2}}]
        report = {"per_seed": [
            {"seed": 11, "mode": mode, "stage_metrics": stages,
             "replay_updates": 2, "memory_capacity": 4, "gradient_updates": 6}
            for mode in ("replay", "recent_only")]}
        result = m.replay_retention(report)
        self.assertAlmostEqual(result["per_seed"][0]["forgetting_delta_mse"], .2)
        self.assertAlmostEqual(result["paired_by_seed"]["11"]["replay_minus_recent_forgetting"], 0.)
