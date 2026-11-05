import unittest
from src import research18_resource_accounting as c
class Tests(unittest.TestCase):
    def test_local_remote(self):
        a=c.cellular(16,4,2,24)
        b=c.cellular(16,4,2,24,remote_edges=8)
        self.assertEqual(a["state_bytes"],256)
        self.assertGreater(b["abstract_cost"],a["abstract_cost"])
        self.assertEqual(a["remote_bytes"],0)
    def test_dense_and_iso(self):
        self.assertEqual(c.dense(3,5)["multiply_accumulates"],15)
        self.assertEqual(len(c.iso_accuracy([{"accuracy":.9},{"accuracy":.7}])),1)
