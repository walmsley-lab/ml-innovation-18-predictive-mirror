import unittest

import jax
import numpy as np

from mirror import forecast_pairs, initial_state, predict_next, preprocess
from src import model, multisheet


class MirrorTests(unittest.TestCase):
    def test_causal_acquisition_order(self):
        order = []
        samples = ((np.zeros((16, 16, 1), np.float32) + i, 10 * (i + 1))
                   for i in range(3))
        def gen():
            for pair in samples:
                order.append("read")
                yield pair
        def predictor(state, current):
            order.append("predict")
            return state, current
        ticks = iter((12, 15, 22, 25, 32, 35))
        def clock():
            return next(ticks)
        records = list(forecast_pairs(gen(), None, predictor, clock=clock))
        self.assertEqual(order, ["read", "predict", "read", "predict", "read", "predict"])
        self.assertEqual(len(records), 2)
        self.assertTrue(all(x["prediction_committed_ns"] < x["frame_arrived_ns"] for x in records))

    def test_recurrent_and_two_sheet_step_are_finite(self):
        image = np.zeros((16, 16, 1), np.float32)
        for architecture, module in (("recurrent", model), ("multisheet", multisheet)):
            params = module.init(jax.random.PRNGKey(2))
            _, forecast = predict_next(params, architecture, initial_state(architecture, 16), image)
            self.assertEqual(forecast.shape, image.shape)
            self.assertTrue(np.isfinite(forecast).all())

    def test_preprocessing_center_crop_gray(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv-python not installed")
        image = np.zeros((12, 20, 3), np.uint8)
        image[:, 4:16, 1] = 255
        result = preprocess(image, 16, cv2)
        self.assertEqual(result.shape, (16, 16, 1))
        self.assertTrue(np.all((0 <= result) & (result <= 1)))


if __name__ == "__main__":
    unittest.main()
