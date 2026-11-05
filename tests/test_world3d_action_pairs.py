import tempfile
import unittest
import numpy as np
from src import world3d_action_pairs as pairs
from experiments.world3d_action_pairs import run


class CounterfactualActionsTests(unittest.TestCase):
    def test_identical_history_and_different_future(self):
        a,b=pairs.pair(21,context=4,horizon=3,size=24)
        np.testing.assert_array_equal(a["rgb"][:4],b["rgb"][:4])
        np.testing.assert_array_equal(a["centers_world"],b["centers_world"])
        self.assertGreater(np.mean(abs(a["rgb"][4:]-b["rgb"][4:])),0)
        self.assertEqual(a["camera_actions"].shape,(6,2))

    def test_zero_response_is_detected(self):
        a,b=pairs.pair(21,4,3,24)
        same=a["rgb"][4:]
        report=pairs.scores(a,b,same,same,context=4)
        self.assertEqual(report["model_action_l1"],0)
        self.assertGreater(report["world_action_l1"],0)

    def test_frozen_checkpoint_is_optional_but_smoke_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            a=run(out=folder+"/world.json",seeds=(1,),context=3,horizon=2,size=16)
            self.assertIsNone(a["checkpoint_sha256"])
            b=run(out=folder+"/smoke.json",seeds=(1,),context=3,horizon=2,
                  size=16,untrained_smoke=True)
            self.assertTrue(b["untrained_smoke"])
            self.assertIn("model_action_l1",b["results"][0]["scores"])


if __name__=="__main__":
    unittest.main()
