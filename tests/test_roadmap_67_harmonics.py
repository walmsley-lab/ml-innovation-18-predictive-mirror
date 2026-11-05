import unittest
import numpy as np
from experiments import roadmap_67_harmonics as m
from experiments import roadmap_62_connectome_contract as c


class HarmonicInterventionTests(unittest.TestCase):
    def test_energy_matching_and_zero_sham(self):
        a,_=c.toy()
        eigen,basis=m.basis(a,3)
        self.assertTrue((eigen>0).all())
        state=np.arange(len(a),dtype=float)
        proj=m.modes(state,basis,11)
        self.assertAlmostEqual(np.linalg.norm(proj["low"]),
                               np.linalg.norm(proj["matched_random"]),places=6)
        self.assertEqual(np.linalg.norm(proj["sham"]),0.)

    def test_seeded_effects_scoring_only(self):
        result=m.run(None,seeds=(11,),horizon=3)
        self.assertTrue(result["graph_symmetrized_for_spectrum"])
        sham=[x for x in result["rows"] if x["arm"]=="sham"]
        self.assertTrue(all(x["accuracy_damage_mse"]==0 for x in sham))


if __name__=="__main__":
    unittest.main()
