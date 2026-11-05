import unittest
from src import research18_closed_loop_control as g
class Tests(unittest.TestCase):
    def test_no_future_access(self):
        ep=g.trajectory(5)
        self.assertIsNone(ep["observations"][9])
        self.assertEqual(len(ep["evaluator_only"]),20)
        out=g.evaluate(g.reactive_policy,[ep],10)
        self.assertIn("hit_rate",out)
    def test_model_beats_reactive_on_linear_world(self):
        worlds=[g.trajectory(i) for i in range(10)]
        a=g.evaluate(g.reactive_policy,worlds)
        b=g.evaluate(g.velocity_policy,worlds)
        self.assertLessEqual(b["mean_error"],a["mean_error"])
