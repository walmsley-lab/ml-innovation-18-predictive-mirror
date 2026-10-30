import tempfile
import unittest

import jax
import numpy as np

from src import multisheet, multisheet_no_feedback
from src.world import batch
from train import train
from evaluate import evaluate


class FeedbackControlTests(unittest.TestCase):
    def test_wrapper_is_identical_to_cut_feedback_and_has_same_nominal_size(self):
        key = jax.random.PRNGKey(7)
        p = multisheet.init(key)
        q = multisheet_no_feedback.init(key)
        self.assertEqual(sum(v.size for v in jax.tree.leaves(p)),
                         sum(v.size for v in jax.tree.leaves(q)))
        x, a = batch([123], steps=5)
        np.testing.assert_array_equal(
            multisheet_no_feedback.predict(q, x[0], a[0]),
            multisheet.predict(p, x[0], a[0], feedback_strength=0.0))

    def test_train_from_scratch_and_evaluate_four_motion_regimes(self):
        with tempfile.TemporaryDirectory() as root:
            result = train(seed=42, steps=2, batch_size=2,
                           architecture="multisheet_no_feedback", output=root)
            self.assertEqual(result["architecture"], "multisheet_no_feedback")
            result = evaluate(multisheet_no_feedback.init(jax.random.PRNGKey(42)),
                              [30_000_000], architecture="multisheet_no_feedback")
            self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
