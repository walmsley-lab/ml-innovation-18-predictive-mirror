import json
import unittest
from pathlib import Path
from experiments.developmental_capabilities import validate


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.charter = json.loads(Path("experiments/developmental_capabilities.json").read_text())

    def test_specification_complete(self):
        result = validate(self.charter)
        self.assertGreaterEqual(result["capabilities"], 5)
        self.assertFalse(result["confirmatory_results"])

    def test_duplicate_and_future_input_rejected(self):
        self.charter["capabilities"].append(dict(self.charter["capabilities"][0]))
        with self.assertRaises(ValueError):
            validate(self.charter)
        self.charter["capabilities"].pop()
        self.charter["capabilities"][0]["observations"] = ["future world truth"]
        with self.assertRaises(ValueError):
            validate(self.charter)


if __name__ == "__main__":
    unittest.main()
