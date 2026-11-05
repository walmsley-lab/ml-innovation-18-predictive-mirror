"""Paired curriculum order, independent splits, and one small training run."""
import tempfile
import unittest
from collections import Counter
import numpy as np
from experiments import development_23_curriculum as exp


class Development23Tests(unittest.TestCase):
    def test_equal_multiset_and_replay_locations(self):
        for steps in (4, 10, 43):
            arms = exp.schedules(steps, 17)
            self.assertEqual(Counter(arms["staged"]), Counter(arms["mixed"]))
            self.assertEqual([i for i,x in enumerate(arms["staged"]) if x[0]=="replay"],
                             [i for i,x in enumerate(arms["mixed"]) if x[0]=="replay"])

    def test_split_and_world_shapes(self):
        train, val, test = exp.episodes(12, 4, 2, 20)
        self.assertEqual(val.shape, (4, 6, 20, 20, 3))
        self.assertEqual(test.shape, val.shape)
        self.assertFalse(np.array_equal(val, test))
        self.assertEqual(len(train["replay"]), 1)

    def test_four_step_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            r = exp.run(root, seeds=(3,), steps=4, context=3, horizon=2,
                        size=20, channels=2, ticks=1)
            self.assertEqual(len(r["results"]), 2)
            self.assertTrue(r["training_episode_multiset_equal"])
            self.assertTrue(all(row["best_step"] <= 4 for row in r["results"]))


if __name__ == "__main__":
    unittest.main()
