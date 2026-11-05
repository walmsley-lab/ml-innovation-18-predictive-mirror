import unittest
import numpy as np
from experiments import roadmap_65_lesions as m
from experiments import roadmap_62_connectome_contract as c


class LesionTests(unittest.TestCase):
    def test_lesion_preserves_original_and_deletes_edges(self):
        graph,_=c.toy()
        intact=graph.copy()
        lesion=m.lesion(graph,2)
        self.assertEqual(lesion[:,2].sum(),0)
        self.assertEqual(lesion[2,:].sum(),0)
        np.testing.assert_array_equal(graph,intact)

    def test_causal_sham_and_matched_control(self):
        result=m.run(None,seeds=(11,),horizon=3)
        scores={x["arm"]:x for x in result["rows"]}
        self.assertEqual(set(scores),
                         {"sham","highest_degree","nearest_degree","random"})
        self.assertEqual(scores["sham"]["damage_delta_mse"],0.)
        self.assertEqual(scores["sham"]["output_effect_l2"],0.)


if __name__=="__main__":
    unittest.main()
