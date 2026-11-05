import tempfile
import unittest
import numpy as np

from experiments.object_permanence_campaign import (
    dataset,selection_score,contrast_error,paired_bootstrap,run,episode_success)


class CampaignTests(unittest.TestCase):
    def test_disjoint_pairs_and_controls(self):
        train,a=dataset("train",2)
        test,b=dataset("test",2)
        self.assertNotEqual(train[0]["seed"],test[0]["seed"])
        self.assertEqual(a.shape,b.shape)
        truth=a[:,8:]
        self.assertAlmostEqual(selection_score(truth,truth),0)
        self.assertAlmostEqual(contrast_error(truth,truth),0)
        truth_scored=episode_success(truth,train,8)
        self.assertEqual(truth_scored["coverage"],4)
        self.assertAlmostEqual(truth_scored["success_rate"],1)

    def test_bootstrap_ci(self):
        x=paired_bootstrap([1,1,0,1],[0,0,0,0],draws=100)
        self.assertEqual(x["pairs"],4)
        self.assertGreater(x["mean_difference"],0)
        with self.assertRaises(ValueError):
            paired_bootstrap([1],[0])

    def test_one_step_smoke_and_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            result=run(directory,models=("cnn",),seeds=(1,),
                       steps=1,width=4,ticks=1,
                       train_pairs=2,val_pairs=2,test_pairs=2,batch_pairs=1)
            self.assertEqual(len(result["runs"]),1)
            self.assertIn("paired_success_vs_baselines",result["runs"][0])
            self.assertIn("paired_success_vs_history_controls",result["runs"][0])
            self.assertEqual(result["runs"][0]["paired_success_vs_baselines"]["persistence"]["pairs"],2)
            self.assertEqual(result["baselines"]["history_linear"]["event"]["coverage"],4)


if __name__=="__main__":
    unittest.main()
