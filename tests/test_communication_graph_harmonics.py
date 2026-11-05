import tempfile
import unittest
import numpy as np
from src import communication_graph_harmonics as h
from src.graph_skeleton_nca import EDGES,JOINTS
from experiments.communication_graph_harmonics import run


class CommunicationHarmonicTests(unittest.TestCase):
    def setUp(self):
        self.actual=h.adjacency_from_edges(len(JOINTS),EDGES)
        self.layout=h.lattice_adjacency(3,4)

    def test_adjacency_and_spectral_eigenpairs(self):
        matrix=h.laplacian(self.actual)
        values,basis=h.eigenbasis(self.actual,3)
        self.assertTrue(np.allclose(basis@basis.T,np.eye(len(basis)),atol=1e-8))
        self.assertTrue(np.allclose(matrix@basis.T,basis.T*values,atol=1e-8))
        self.assertFalse(np.array_equal(self.actual,self.layout))

    def test_degree_preserving_rewires_and_energy_matching(self):
        changed=h.rewired(self.actual,42,30)
        np.testing.assert_array_equal(changed.sum(axis=0),self.actual.sum(axis=0))
        self.assertFalse(np.array_equal(changed,self.actual))
        values=h.interventions(self.actual,np.random.default_rng(1).normal(
            size=(12,2)),steps=4,rank=3)
        self.assertAlmostEqual(values["low_removed_l2"],
                               values["control_removed_l2"],places=8)
        self.assertGreater(values["low_response_l2"],0)

    def test_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            result=run(directory+"/metrics.json",seeds=(1,),steps=2)
            self.assertEqual(len(result["rows"]),1)
            self.assertIn("degree_preserving_rewire",
                          result["rows"][0]["cases"])


if __name__=="__main__":
    unittest.main()
