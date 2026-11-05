import unittest
import numpy as np
from src import physical_fields as f


class PhysicalFieldsTests(unittest.TestCase):
    def test_constant_laplacian_reflect(self):
        np.testing.assert_allclose(f.laplacian(np.ones((8, 9))), 0)

    def test_diffusion_mass_conserves_and_peak_falls(self):
        a = f.episode(11, "diffusion", steps=30)
        mass = a["observed"].sum(axis=(1, 2))
        np.testing.assert_allclose(mass, mass[0], atol=1e-10)
        self.assertLess(a["observed"][-1].max(), a["observed"][0].max())

    def test_wave_needs_velocity_and_propagates(self):
        a = f.episode(9, "wave", steps=16)
        self.assertEqual(a["states"].shape[1], 2)
        self.assertTrue(np.max(np.abs(a["states"][:, 1])) > 0)
        self.assertFalse(np.allclose(a["observed"][0], a["observed"][3]))

    def test_stability_checks_and_boundary(self):
        with self.assertRaises(ValueError):
            f.episode(kind="diffusion", dt=2, diffusivity=.3)
        with self.assertRaises(ValueError):
            f.episode(kind="wave", dt=2, speed=1)
        u = np.ones((10, 10))
        self.assertLess(f.laplacian(u, boundary="zero")[0, 0], 0)
        self.assertLess(abs(f.laplacian(u, boundary="reflect")).max(), 1e-12)

    def test_repeatability_and_verifier(self):
        a = f.episode(5, "wave", steps=5)
        b = f.episode(5, "wave", steps=5)
        np.testing.assert_array_equal(a["states"], b["states"])
        self.assertLess(f.verify()["diffusion_mass_max_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
