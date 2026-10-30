import unittest

import jax
import numpy as np

from mirror import activity_maps, forecast_pairs, initial_state, predict_next
from src import model, multisheet


class ActivityDemoTests(unittest.TestCase):
    def test_model_activity_maps_have_absolute_stable_scale(self):
        for arch, network in (("recurrent", model), ("multisheet", multisheet)):
            params = network.init(jax.random.PRNGKey(9))
            state = initial_state(arch, 16)
            state, prediction = predict_next(params, arch, state,
                                             np.zeros((16, 16, 1), np.float32))
            a, b = activity_maps(state)
            self.assertEqual(a.shape, (16, 16))
            self.assertEqual(b.shape, (16, 16))
            self.assertTrue(np.all((a >= 0) & (a <= 1)))
            self.assertTrue(np.all((b >= 0) & (b <= 1)))
            self.assertEqual(prediction.shape, (16, 16, 1))

    def test_activity_is_recorded_before_target_frame_arrives(self):
        order = []
        def observations():
            for i in range(3):
                order.append("read")
                yield np.ones((16, 16, 1), np.float32) * i, 100 * (i + 1)
        def predictor(state, current):
            order.append("predict")
            return state, current
        clocks = iter([10, 20, 110, 120, 210, 220])
        records = list(forecast_pairs(observations(), None, predictor, clock=lambda: next(clocks)))
        self.assertEqual(len(records), 2)
        self.assertEqual(order, ["read", "predict", "read", "predict", "read", "predict"])
        self.assertTrue(all(x["prediction_committed_ns"] < x["frame_arrived_ns"] for x in records))
        self.assertIsNone(records[0]["fast_activity"])


if __name__ == "__main__":
    unittest.main()
