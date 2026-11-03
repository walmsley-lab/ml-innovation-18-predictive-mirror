import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.rgb_robust_world import DOMAINS, episode, model_inputs
from experiments.rgb_ood_world import run


class RGBRobustWorldTests(unittest.TestCase):
    def test_deterministic_and_bounded(self):
        for domain in DOMAINS:
            first = episode(12, domain=domain)
            second = episode(12, domain=domain)
            np.testing.assert_array_equal(first["frames"],second["frames"])
            self.assertEqual(first["frames"].shape,(18,32,32,3))
            self.assertEqual(first["camera_actions"].shape,(17,2))
            self.assertEqual(first["occluded"].shape,(18,2))
            self.assertGreaterEqual(first["frames"].min(),0)
            self.assertLessEqual(first["frames"].max(),1)

    def test_rgb_only_contract_excludes_truth_and_actions(self):
        sample = episode(42,domain="ood_camera")
        self.assertEqual(set(model_inputs(sample)),{"frames"})
        self.assertEqual(set(model_inputs(sample,True)),{"frames","camera_actions"})
        self.assertGreater(np.abs(sample["camera_actions"]).sum(),0)

    def test_splits_change_the_world(self):
        simple = episode(77,domain="train")
        novel = episode(77,domain="ood_combined")
        self.assertFalse(np.array_equal(simple["frames"],novel["frames"]))

    def test_cli_writes_manifest_and_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"manifest.json"
            manifest = run(path=path,episodes=2,steps=9,size=24)
            self.assertTrue(path.exists())
            self.assertTrue((path.parent/"examples.npz").exists())
            self.assertNotIn("positions",manifest["model_visible_by_default"])


if __name__ == "__main__":
    unittest.main()
