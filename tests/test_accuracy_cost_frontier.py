import unittest
from src.accuracy_cost_frontier import nondominated


class FrontierTests(unittest.TestCase):
    def setUp(self):
        self.rows=[
            {"model":"strong_fast","protocol":"same","device":"cpu",
             "success_rate":.8,"latency_ms":5.},
            {"model":"weak_slow","protocol":"same","device":"cpu",
             "success_rate":.7,"latency_ms":15.},
            {"model":"accurate_slow","protocol":"same","device":"cpu",
             "success_rate":.9,"latency_ms":30.}]

    def test_distinct_pareto_choices_remain(self):
        result=nondominated(self.rows)
        self.assertFalse(result[1]["pareto_efficient"])
        self.assertTrue(result[0]["pareto_efficient"])
        self.assertTrue(result[2]["pareto_efficient"])

    def test_comparison_fails_with_mismatched_protocols(self):
        self.rows[1]["protocol"]="other"
        with self.assertRaisesRegex(ValueError,"different tasks"):
            nondominated(self.rows)


if __name__=="__main__":
    unittest.main()
