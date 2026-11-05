import unittest
import numpy as np
from src import research18_hidden_rules_world as w
class Tests(unittest.TestCase):
    def test_ambiguous_prefix(self):
        a,b=w.paired_rules(7)
        switch=a["evaluator_only"]["switch_time"]
        np.testing.assert_array_equal(a["observations"][:switch],b["observations"][:switch])
        self.assertFalse(np.array_equal(a["evaluator_only"]["positions"][-1],
                                         b["evaluator_only"]["positions"][-1]))
    def test_visibility_and_actions(self):
        x=w.episode(8,occlude=True)
        self.assertEqual(x["observations"].shape,(20,32,32))
        self.assertEqual(x["model_actions"].shape,(20,2))
