"""The event gallery must emit a correct headless manifest (PNGs optional)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.rgb_event_gallery import run, manifest
from src.rgb_motion_world import episode


class RGBEventGalleryTests(unittest.TestCase):
    def test_manifest_flags_present_and_absent_objects(self):
        records = [episode(321, steps=9, size=24, balls=2, occlusion=True)]
        truth = records[0]["frames"][4:9][None]
        blank = np.zeros_like(truth)
        out = manifest({"truth": truth, "blank": blank}, records,
                       context=4, leads=(1, 3))
        saw_object = False
        for lead_objects in out[0]["leads"].values():
            for entry in lead_objects:
                saw_object = True
                t = entry["models"]["truth"]
                b = entry["models"]["blank"]
                # A perfect forecast finds the object with full mass; a blank one
                # has no mass and no soft centroid (genuine disappearance).
                self.assertTrue(t["hard_found"])
                self.assertGreater(t["mass_fraction"], 0.8)
                self.assertAlmostEqual(b["mass_fraction"], 0.0)
                self.assertIsNone(b["soft_centroid"])
                self.assertFalse(b["hard_found"])
        self.assertTrue(saw_object)

    def test_run_writes_gallery_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run(out=tmp, checkpoint=None, episodes=2, context=4,
                          horizon=3, size=24, balls=1, occlusion=False)
            path = Path(tmp) / "gallery.json"
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(set(data["models"]),
                             {"persistence", "constant_velocity"})
            self.assertEqual(len(data["episodes"]), 2)


if __name__ == "__main__":
    unittest.main()
