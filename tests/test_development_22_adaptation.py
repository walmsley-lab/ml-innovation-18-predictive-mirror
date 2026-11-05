"""Known-answer and causality tests for sensory adaptation."""
import unittest
import numpy as np
from experiments import development_22_adaptation as exp


class Development22Tests(unittest.TestCase):
    def test_causal_prefix_and_shapes(self):
        frames, _, _ = exp.world(11)
        a = exp.channels(frames[:10])
        b = exp.channels(frames)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y[:10], atol=1e-7)
            self.assertEqual(y.shape, frames.shape)

    def test_stationary_content_not_erased(self):
        r = exp.run(11, out=None)
        self.assertLess(r["normalized_contrast_cv"], r["raw_contrast_cv"])
        self.assertGreater(r["static_retained_contrast"], .25)
        self.assertLess(abs(r["change_only_static_contrast"]), .15)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            exp.channels(np.zeros((3, 8, 8)))
        with self.assertRaises(ValueError):
            exp.channels(np.zeros((3, 8, 8, 3)), decay=1.0)


if __name__ == "__main__":
    unittest.main()
