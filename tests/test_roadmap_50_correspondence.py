import unittest
import numpy as np
from experiments import roadmap_50_correspondence as m
from experiments import development_24_correspondence as reference


class CorrespondenceTests(unittest.TestCase):
    def test_causal_prefix(self):
        rgb,_,_=reference.world(11)
        np.testing.assert_allclose(m.track(rgb)[:5],m.track(rgb[:5]))
        self.assertEqual(m.track(rgb).shape,(len(rgb),2,2))

    def test_one_to_one_assignments(self):
        p=np.array([[5.,5.],[18.,5.]])
        found=np.array([[6.,5.],[17.,5.]])
        self.assertTrue(list(m.proposals(p,np.zeros_like(p),found)))
        with self.assertRaises(ValueError):
            m.track([np.zeros((2,2,3))])

    def test_heldout_scores(self):
        r=m.run(None,seeds=(91000,))
        self.assertEqual(len(r["rows"]),4)
        self.assertTrue(all(x["scores"]["visible_mae_px"]>=0 for x in r["rows"]))


if __name__=="__main__":
    unittest.main()
