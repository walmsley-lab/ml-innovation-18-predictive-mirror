"""Test geometry-resonance dynamics and their causal sensory bridge."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import geometry_automata as world
from src import geometry_harmonics as h
from src import geometry_perceiver as perceiver
from experiments.geometry_harmonics import study
from experiments.geometry_automata import run as train_local


class GeometryHarmonicsTests(unittest.TestCase):
    def test_graph_eigenmodes_and_relabeling(self):
        xy, edges = world.generate(11, 12, "irregular")
        lap = h.laplacian(xy, edges)
        self.assertTrue(np.allclose(lap, lap.T))
        self.assertTrue(np.allclose(lap.sum(axis=0), 0., atol=1e-12))
        omega, phi = h.modes(xy, edges)
        self.assertGreater(omega.min(), 0.)
        self.assertTrue(np.all(np.diff(omega) >= -1e-12))
        np.testing.assert_allclose(phi.T @ phi, np.eye(len(xy)), atol=1e-12)
        perm = np.random.default_rng(7).permutation(len(xy))
        inverse = np.argsort(perm)
        translated = h.laplacian(xy[perm], inverse[edges])
        np.testing.assert_allclose(translated, lap[np.ix_(perm, perm)], atol=1e-10)

    def test_linear_superposition_not_nonlinear_mixing(self):
        xy, edges = world.generate(12, 13, "ring")
        frequencies, basis = h.modes(xy, edges)
        forcing_a = h.tone(320, .06, frequencies[1], basis[:, 1], .5)
        forcing_b = h.tone(320, .06, frequencies[2], basis[:, 2], .5)
        a, _ = h.simulate(xy, edges, forcing_a, dt=.06)
        b, _ = h.simulate(xy, edges, forcing_b, dt=.06)
        together, _ = h.simulate(xy, edges, forcing_a+forcing_b, dt=.06)
        linear = float(np.max(np.abs(together-a-b)))
        self.assertLess(linear, 1e-10)
        an, _ = h.simulate(xy, edges, forcing_a, dt=.06, cubic=3.)
        bn, _ = h.simulate(xy, edges, forcing_b, dt=.06, cubic=3.)
        jointn, _ = h.simulate(xy, edges, forcing_a+forcing_b, dt=.06, cubic=3.)
        self.assertGreater(float(np.mean((jointn-an-bn)**2)), 1e-10)

    def test_broadband_drive_is_reproducible_and_external(self):
        excitation = h.broadband(23, 90, 12, .25)
        np.testing.assert_array_equal(excitation, h.broadband(23, 90, 12, .25))
        self.assertEqual(np.sum(np.any(excitation != 0, axis=0)), 1)
        xy, edges = world.generate(23, 12)
        driven, velocity = h.simulate(xy, edges, excitation)
        silent, _ = h.simulate(xy, edges, np.zeros_like(excitation))
        np.testing.assert_allclose(silent, 0.)
        self.assertGreater(np.max(np.abs(driven)), 0.)
        self.assertTrue(np.isfinite(velocity).all())

    def test_phase_chart_and_forced_recovery(self):
        xy, edges = world.generate(41, 13, "branch")
        omega, phi = h.modes(xy, edges)
        excitation = h.tone(720, .055, omega[1], phi[:, 1], .30)
        reference, velocity = h.simulate(xy, edges, excitation)
        helix, angles = h.phase_trajectory(reference, velocity, omega[1], phi[:, 1])
        self.assertEqual(helix.shape, (721, 3))
        self.assertTrue(np.isfinite(angles).all())
        self.assertGreater(abs(angles[-1] - angles[360]), 2*np.pi)
        perturb = np.zeros(len(xy))
        perturb[2] = .25
        disturbed, _ = h.simulate(xy, edges, excitation, shock=(360, perturb))
        drift = np.sqrt(np.mean((disturbed-reference)**2, axis=1))
        self.assertGreater(drift[362:380].mean(), 0.)
        self.assertLess(drift[-35:].mean(), drift[362:400].mean())

    def test_real_mode_scan_produces_evidence(self):
        evidence = study(11, nodes=12, steps=260, kind="irregular")
        self.assertGreater(evidence["resonant_modal_rms"], 0.)
        self.assertTrue(np.isfinite(evidence["broadband_modal_peak_rad_per_s"]))
        self.assertLess(evidence["linear_superposition_rmse"], 1e-10)
        self.assertGreater(evidence["nonlinear_superposition_rmse"], 0.)

    def test_harmonic_sensors_do_not_expose_truth_or_future(self):
        record = h.sensor_record(71, nodes=13, kind="ring", steps=8)
        self.assertEqual(record["observed"].shape, (9, 13, 2))
        self.assertEqual(record["forces"].shape, (8, 13, 2))
        self.assertLess(record["sensor_mask"][0].sum(), 13)
        self.assertTrue(np.all(record["observed"] *
                               (1-record["sensor_mask"]) == 0.))
        params = perceiver.init(jax.random.PRNGKey(71), 8)
        values = [jnp.asarray(record[k]) for k in
                  ("observed", "sensor_mask", "forces", "adjacency")]
        forecast = np.asarray(perceiver.predict(params, *values))
        changed = values[0].at[1:].set(100.)
        next_forecast = np.asarray(perceiver.predict(
            params, changed, *values[1:]))
        np.testing.assert_array_equal(next_forecast[0], forecast[0])

    def test_shared_cellular_rule_harmonic_training_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = train_local(temporary, seeds=(11,), steps=2,
                                 channels=8, medium="harmonic")
            self.assertEqual(result["medium"], "harmonic")
            self.assertEqual(set(result["reports"][0]["arms"]),
                             {"messages", "no_messages"})
            self.assertTrue((Path(temporary) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
