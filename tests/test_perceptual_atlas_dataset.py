"""CPU-only checks for paired geometric observations and dataset provenance."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import perceptual_atlas_dataset as atlas


class PerceptualAtlasDatasetTests(unittest.TestCase):
    def test_reproducible_paired_observations(self):
        a = atlas.generate(seed=190, episodes=4, frames=5, size=16)
        b = atlas.generate(seed=190, episodes=4, frames=5, size=16)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(a["frames_dark"].shape, (4, 5, 16, 16, 3))
        self.assertEqual(a["latent_intensity"].shape, (4, 5, 16, 16))
        self.assertEqual(a["cue_hz"].shape, (4, 5))
        self.assertTrue(np.any(a["frames_dark"] != a["frames_bright"]))
        self.assertTrue(np.all((a["latent_intensity"] >= 0) &
                               (a["latent_intensity"] <= 1)))

    def test_light_changes_only_the_renderer(self):
        params = atlas.scene(15, mode="chamber", coupling=0.7)
        intensity, hue = atlas.field(24, 1.2, params)
        alternate, alternate_hue = atlas.field(24, 1.2, {**params, "light":1.0})
        np.testing.assert_array_equal(intensity, alternate)
        np.testing.assert_array_equal(hue, alternate_hue)
        self.assertFalse(np.array_equal(atlas.render(intensity, hue, 0.1),
                                        atlas.render(intensity, hue, 0.9)))

    def test_coupling_changes_field_when_pitch_changes(self):
        base = atlas.scene(16, mode="fold", coupling=1, pitch=0.1)
        other = {**base, "pitch":0.9}
        self.assertFalse(np.array_equal(atlas.field(24, 1.0, base)[0],
                                        atlas.field(24, 1.0, other)[0]))
        zero = {**base, "coupling":0}
        np.testing.assert_array_equal(atlas.field(24, 1.0, zero)[0],
                                      atlas.field(24, 1.0, {**zero, "pitch":0.9})[0])

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError): atlas.scene(1, mode="missing")
        with self.assertRaises(ValueError): atlas.episode(1, "fold", frames=1)
        with self.assertRaises(ValueError): atlas.generate(seed=-1)

    def test_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dataset.npz"
            report = atlas.run(str(output), episodes=2, frames=3, size=8)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".json").exists())
            self.assertEqual(report["shape"], [2, 3, 8, 8, 3])
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files),
                    {"frames_dark", "frames_bright", "latent_intensity", "cue_hz"})
            with self.assertRaises(FileExistsError):
                atlas.run(str(output), episodes=2, frames=3, size=8)


if __name__ == "__main__":
    unittest.main()
