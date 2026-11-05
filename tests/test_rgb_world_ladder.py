import tempfile
import unittest
import numpy as np
from src import rgb_world_ladder as world
from experiments.rgb_world_ladder import run


class WorldLadderTests(unittest.TestCase):
    def test_reproducible_and_truth_separated(self):
        for domain in world.DOMAINS:
            a=world.episode(123,steps=18,size=32,balls=2,domain=domain)
            b=world.episode(123,steps=18,size=32,balls=2,domain=domain)
            np.testing.assert_array_equal(a["frames"],b["frames"])
            self.assertEqual(a["frames"].shape,(18,32,32,3))
            self.assertEqual(a["visibility_reason"].shape,(18,2))
            self.assertEqual(set(world.model_inputs(a)),{"frames"})
            self.assertEqual(set(world.model_inputs(a,True)),
                             {"frames","camera_actions"})

    def test_split_disjoint_and_scripted_occluder(self):
        train=world.split("train",1,domain="occlusion_long")[0]
        test=world.split("test",1,domain="occlusion_long")[0]
        self.assertNotEqual(train["seed"],test["seed"])
        self.assertTrue(np.any(test["visibility_reason"]=="barrier"))

    def test_baselines_and_event_score_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            report=run(folder+"/results.json",domains=("collision","occlusion_long"),
                       episodes=2,context=4,horizon=3,size=24,balls=2)
            self.assertEqual(set(report["results"]),{"collision","occlusion_long"})
            self.assertIn("persistence",report["results"]["collision"]["baselines"])


if __name__=="__main__":
    unittest.main()
