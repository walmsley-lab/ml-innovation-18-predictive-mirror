import tempfile
import unittest
from experiments import roadmap_55_scene_campaign as m


class SceneCampaignTests(unittest.TestCase):
    def test_invalid_seeds_and_missing_checkpoint_fail(self):
        with self.assertRaises(ValueError):
            m.run(None,seeds=(11,11))
        with self.assertRaises(ValueError):
            m.run(None,seeds=(11,12),checkpoints={"nca":"x"})

    def test_bundle_is_not_global_leaderboard(self):
        with tempfile.TemporaryDirectory() as directory:
            r=m.run(directory,seeds=(91000,91001))
            self.assertEqual(len(r["parts"]),9)
            self.assertEqual(r["trained_rgb_status"],"skipped_missing_checkpoints")
            self.assertTrue(all(p["source_sha256"] for p in r["parts"]))
            self.assertIn("No comparison",r["constraints"])


if __name__=="__main__":
    unittest.main()
