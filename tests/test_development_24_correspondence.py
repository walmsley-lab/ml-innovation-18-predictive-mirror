import unittest
import numpy as np
from experiments import development_24_correspondence as exp


class Development24Tests(unittest.TestCase):
    def test_missingness_and_identity_are_evaluator_only(self):
        rgb, truth, visible = exp.world(13)
        self.assertEqual(rgb.shape[0], len(truth))
        self.assertEqual(visible.shape, (len(rgb), 2))
        self.assertEqual(len(exp.detections(rgb[0])), 2)
        self.assertEqual(len(exp.detections(rgb[4])), 1)
        self.assertEqual(exp.track(rgb).shape, truth.shape)

    def test_one_to_one_and_causal_prefix(self):
        result = exp.match(np.array([[2., 0.], [4., 0.]]),
                           np.array([[2.1, 0.], [4.1, 0.]]))
        self.assertEqual(len(result), 2)
        a, _, _ = exp.world()
        np.testing.assert_allclose(exp.track(a)[:4], exp.track(a[:4]))

    def test_controls_and_invalid_mode(self):
        result = exp.run(out=None)
        self.assertTrue(np.isfinite(result["scores"]["coast"]["visible_mae_px"]))
        self.assertIn("nearest", result["scores"])
        with self.assertRaises(ValueError):
            exp.track(exp.world()[0], "oracle")


if __name__ == "__main__":
    unittest.main()
