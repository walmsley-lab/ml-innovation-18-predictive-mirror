import unittest
from src import research18_substrate_mapping as m
class Tests(unittest.TestCase):
    def test_hops(self):
        self.assertEqual(m.hop_distance((0,0),(2,3)),5)
        x=m.map_communication(4,[(0,1),(0,3)],columns=2,
                              channels=1,word_bytes=1,local_capacity_bytes=2)
        self.assertEqual(x["state_bytes"],4)
        self.assertEqual(x["offchip_spill_floor_bytes"],2)
        self.assertEqual(x["weighted_hop_bytes"],3)
        self.assertEqual(x["credibility"],"analytic_not_hardware_measured")
    def test_invalid(self):
        with self.assertRaises(ValueError):m.positions(0,2)
