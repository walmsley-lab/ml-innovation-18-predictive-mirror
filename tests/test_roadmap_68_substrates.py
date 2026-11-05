import unittest
from experiments import roadmap_68_substrates as m


class SubstrateTests(unittest.TestCase):
    def test_same_node_count_and_explicit_comm_cost(self):
        arms=m.choices()
        self.assertEqual(len(arms),4)
        self.assertTrue(all(a.shape==(12,12) for a in arms.values()))
        self.assertEqual(m.communication_cost(arms["uncoupled"],3)
                         ["edge_reads_proxy"],0)

    def test_matched_weights_and_heldout(self):
        report=m.run(None,seeds=(11,),horizon=3)
        self.assertEqual(len(report["rows"]),4)
        self.assertEqual({r["parameters"] for r in report["rows"]},{3})
        self.assertTrue(all(r["communication"]["hardware_energy_joules"] is None
                            for r in report["rows"]))


if __name__=="__main__":
    unittest.main()
