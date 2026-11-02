"""Curriculum order is the only difference in exposure between arms."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.rgb_developmental_curriculum import (
    STAGES, curriculum_orders, make_batches, separate_splits, run)


class DevelopmentalCurriculumTests(unittest.TestCase):
    def test_exactly_identical_exposure_multisets(self):
        orders, labels = curriculum_orders(12, 17)
        self.assertEqual(sorted(orders["staged"].tolist()),
                         sorted(orders["shuffled"].tolist()))
        staged_labels = [int(labels[i]) for i in orders["staged"]]
        shuffled_labels = [int(labels[i]) for i in orders["shuffled"]]
        self.assertEqual(sorted(staged_labels), sorted(shuffled_labels))
        self.assertEqual(staged_labels, sorted(staged_labels))
        self.assertNotEqual(staged_labels, shuffled_labels)
        self.assertEqual(len(STAGES), 3)

    def test_deterministic_training_and_separate_splits(self):
        a, labels = make_batches(3, 1, 4, 2, 16)
        b, same_labels = make_batches(3, 1, 4, 2, 16)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(labels, same_labels)
        validation, test = separate_splits(4, 2, 16, 2)
        for stage in STAGES:
            self.assertEqual(validation[stage].shape, (2, 6, 16, 16, 3))
            self.assertFalse(np.array_equal(validation[stage], test[stage]))

    def test_invalid_stage_counts(self):
        with self.assertRaises(ValueError):
            curriculum_orders(4, 1)

    def test_smoke_with_step_zero_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(folder, seeds=(3,), steps=3, batch_size=1,
                         heldout=1, context=4, horizon=2,
                         size=16, ticks=1, channels=4)
            arms = result["results"][0]["arms"]
            self.assertEqual(set(arms), {"staged", "shuffled"})
            self.assertTrue(result["same_training_multiset_per_arm"])
            for arm in arms.values():
                self.assertLessEqual(arm["selected_validation_mse"],
                                     arm["step_zero_validation_mse"])
                self.assertEqual(set(arm["test_by_scenario"]), set(STAGES))
            self.assertTrue((Path(folder)/"seed-3-staged.npz").exists())
            self.assertTrue((Path(folder)/"seed-3-shuffled-example.npz").exists())


if __name__ == "__main__":
    unittest.main()
