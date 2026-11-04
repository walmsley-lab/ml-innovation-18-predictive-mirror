"""Check matched-energy nulls, no-future leakage and forecast parity."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca
from src import rgb_harmonics as hm
from src import rgb_mode_interventions as probe
from experiments.rgb_mode_interventions import run


class ModeInterventionTests(unittest.TestCase):
    def test_energy_match_and_constant_component_untouched(self):
        _, modes = hm.lattice_modes(8, 8, 5)
        null = probe.random_modes(8, 8, len(modes), 9)
        rng = np.random.default_rng(8)
        hidden = jnp.asarray(rng.normal(size=(2, 8, 8, 3)).astype(np.float32))
        low = np.asarray(probe.perturb(hidden, modes, null, "suppress_low_modes"))
        rand = np.asarray(probe.perturb(hidden, modes, null, "matched_random"))
        reference = np.asarray(hidden)
        np.testing.assert_allclose(np.square(reference - low).sum(axis=(1, 2)),
                                   np.square(reference - rand).sum(axis=(1, 2)),
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(low.mean(axis=(1, 2)),
                                   reference.mean(axis=(1, 2)), atol=1e-6)
        np.testing.assert_allclose(rand.mean(axis=(1, 2)),
                                   reference.mean(axis=(1, 2)), atol=1e-6)

    def test_mode_removal_targets_constructed_eigenmode(self):
        _, basis = hm.lattice_modes(8, 8, 4)
        null = probe.random_modes(8, 8, len(basis), 1)
        hidden = jnp.asarray(basis[0][None, :, :, None], jnp.float32)
        result = probe.perturb(hidden, basis, null, "suppress_low_modes")
        np.testing.assert_allclose(np.asarray(result), 0, atol=1e-5)

    def test_intact_parity_and_late_intervention(self):
        rng = np.random.default_rng(3)
        x = rng.random((1, 7, 8, 8, 3), dtype=np.float32)
        params = nca.init(jax.random.PRNGKey(2), 4)
        # Make readout nonzero so post-context hidden interventions can affect RGB.
        params = dict(params, rgb_readout=jnp.ones((4, 3), jnp.float32) * .05)
        _, basis = hm.lattice_modes(8, 8, 4)
        null = probe.random_modes(8, 8, len(basis), 2)
        prefix = jnp.asarray(x[:, :4])
        intact = np.asarray(probe.forecast(params, prefix, basis, null, 3, 1))
        np.testing.assert_allclose(intact,
            np.asarray(nca.rollout(params, prefix, 3, 1)), atol=1e-6)
        changed = np.asarray(probe.forecast(
            params, prefix, basis, null, 3, 1, "suppress_low_modes", 2))
        np.testing.assert_array_equal(changed[:, :2], intact[:, :2])
        altered = x.copy()
        altered[:, 4:] = 0
        np.testing.assert_array_equal(
            changed, np.asarray(probe.forecast(
                params, jnp.asarray(altered[:, :4]), basis, null,
                3, 1, "suppress_low_modes", 2)))

    def test_runner_on_zero_readout_checkpoint(self):
        params = nca.init(jax.random.PRNGKey(3), 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "params.npz"
            np.savez_compressed(checkpoint, **jax.tree.map(np.asarray, params))
            report = run(checkpoint, Path(directory) / "report",
                         episodes=1, context=4, horizon=2, size=16,
                         ticks=1, mode_count=4, intervene_at=1)
            self.assertEqual(set(report["interventions"]), set(probe.MODES))
            for result in report["interventions"].values():
                self.assertEqual(result["forecast_mean_absolute_change"], 0.0)
            # A zero-readout checkpoint maps hidden state to nothing, so the
            # positive control (wiping hidden memory) must register no change
            # and no lead may be flagged 'live': every ~0 mode-suppression delta
            # is uninformative (a dead or merely insensitive measurement), never
            # a causal null.
            measurement = report["measurement"]
            self.assertIn("clip_rail_saturation", measurement)
            self.assertIn("input_independence", measurement)
            wipe = measurement["positive_control_response"]["wipe_hidden_memory"]
            for lead, change in wipe.items():
                self.assertEqual(change, 0.0)
            self.assertNotIn("live", set(measurement["measurement_verdict"].values()))


if __name__ == "__main__":
    unittest.main()
