import tempfile
import unittest
import numpy as np
from src import gaussian_correspondence as evidence
from experiments.gaussian_correspondence import run


class GaussianCorrespondenceTests(unittest.TestCase):
    def test_permutation_is_not_identity_loss(self):
        rows=[{"frame":0,"objects":{0:{"slot":0,"error_px":1.0,"localized":True}},
               "partial_objects":0,"hidden_objects":0},
              {"frame":1,"objects":{0:{"slot":1,"error_px":1.0,"localized":True}},
               "partial_objects":0,"hidden_objects":0}]
        r=evidence.summarize(rows)
        self.assertEqual(r["raw_slot_switches"],1)
        self.assertEqual(r["localized_permutation_only_switches"],1)
        self.assertEqual(r["visible_identity_failures"],0)

    def test_wrong_slot_location_is_real_failure(self):
        rows=[{"frame":0,"objects":{0:{"slot":0,"error_px":12.0,"localized":False}},
               "partial_objects":0,"hidden_objects":0}]
        self.assertEqual(evidence.summarize(rows)["visible_identity_failures"],1)

    def test_partial_occlusion_and_smoke(self):
        scene=evidence.partial_episode(11,steps=9,size=32)
        ratios=np.stack([evidence.visible_fractions(scene,t) for t in range(9)])
        self.assertTrue(((ratios>0)&(ratios<.98)).any())
        with tempfile.TemporaryDirectory() as path:
            result=run(path+"/metrics.json",untrained_smoke=True,
                       seeds=(11,),steps=6,size=24)
            self.assertEqual(len(result["rows"]),3)


if __name__=="__main__":
    unittest.main()
