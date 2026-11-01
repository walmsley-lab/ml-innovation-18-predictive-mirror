import unittest

import numpy as np

from src.baselines import (constant_velocity_forecast, optical_flow_forecast,
                           persistence_forecast)


def translated_square(shift):
    frame = np.zeros((24, 24, 3), np.float32)
    y, x = 8, 5 + shift
    frame[y:y + 5, x:x + 5] = (1.0, 0.2, 0.05)
    return frame


class RGBBaselineTests(unittest.TestCase):
    def test_persistence_repeats_last_frame(self):
        context = np.stack([translated_square(0), translated_square(1)])
        forecast = persistence_forecast(context, 3)
        self.assertEqual(forecast.shape, (3, 24, 24, 3))
        np.testing.assert_array_equal(forecast[0], context[-1])
        np.testing.assert_array_equal(forecast[-1], context[-1])

    def test_constant_velocity_warps_rgb_scene(self):
        context = np.stack([translated_square(0), translated_square(1)])
        forecast = constant_velocity_forecast(context, 2, max_shift=3)
        np.testing.assert_allclose(forecast[0], translated_square(2), atol=0)
        np.testing.assert_allclose(forecast[1], translated_square(3), atol=0)

    @unittest.skipUnless(__import__('importlib').util.find_spec('cv2'),
                         'OpenCV optional')
    def test_optical_flow_returns_rgb_horizon(self):
        context = np.stack([translated_square(0), translated_square(1)])
        forecast = optical_flow_forecast(context, 2)
        self.assertEqual(forecast.shape, (2, 24, 24, 3))
        self.assertTrue(np.isfinite(forecast).all())


if __name__ == '__main__':
    unittest.main()
