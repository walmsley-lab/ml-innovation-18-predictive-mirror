import unittest
from experiments import roadmap_69_topology_transfer as m


class TopologyTransferTests(unittest.TestCase):
    def test_frozen_transfer_and_target_supervision_distinction(self):
        r=m.run(None,seeds=(11,),horizon=3)
        self.assertEqual(len(r["rows"]),5)
        labels={x["arm"]:x for x in r["rows"]}
        self.assertEqual(len(labels["transferred_weights_target_graph"]["coefficients"]),3)
        self.assertFalse(labels["transferred_weights_target_graph"]["target_train_supervision"])
        self.assertTrue(labels["target_supervised_reference"]["target_train_supervision"])

    def test_seeded_reproducibility(self):
        a=m.run(None,seeds=(11,),horizon=3)
        b=m.run(None,seeds=(11,),horizon=3)
        self.assertEqual(a["rows"],b["rows"])


if __name__=="__main__":
    unittest.main()
