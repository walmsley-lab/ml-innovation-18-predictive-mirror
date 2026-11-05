import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments import roadmap_62_connectome_contract as m


class ConnectomeContractTests(unittest.TestCase):
    def test_source_sha_and_directed_graph(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=m.synthetic_artifact(d,seed=11)
            graph,ids,meta=m.load(a,b)
            self.assertEqual(len(ids),12)
            self.assertEqual(meta["source_type"],"synthetic")
            self.assertTrue(m.summarize(graph,meta)["directed"])
            Path(a).write_bytes(b"tampered")
            with self.assertRaises(ValueError):m.load(a,b)

    def test_reject_missing_nodes_and_negative_weights(self):
        a,ids=m.toy()
        a[0,1]=-1
        with self.assertRaises(ValueError):m.validate(a,ids)
        with self.assertRaises(ValueError):m.validate(np.eye(5),np.arange(4))


if __name__=="__main__":
    unittest.main()
