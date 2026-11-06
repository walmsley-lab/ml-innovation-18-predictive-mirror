import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from experiments.developmental_interaction_path import DevelopmentalSheet
from experiments import developmental_evidence as evidence
from src import developmental_jax


class DevelopmentalEvidenceTests(unittest.TestCase):
    def test_strict_manifest_and_unknown_kind_fail_closed(self):
        config = json.loads(Path("experiments/akash-developmental-smoke.json").read_text())
        self.assertEqual(evidence.validate(config), 4)
        config["kind"] = "unknown"
        with self.assertRaises(ValueError):
            evidence.validate(config)

    def test_jax_step_matches_numpy_reference(self):
        rng = np.random.default_rng(4)
        reference = DevelopmentalSheet(sheet=4, channels=7, seed=11, plasticity=.04)
        frame = rng.random((16, 16, 3), dtype=np.float32)
        action = np.asarray((.04, .012), np.float32)
        initial = {"phase": reference.phase.copy(),
                   "connections": reference.connections.copy()}
        expected = reference.observe(frame, action)
        state = {"phase": jnp.asarray(initial["phase"]),
                 "hidden": jnp.asarray(np.zeros_like(reference.hidden)),
                 "trace": jnp.asarray(np.zeros_like(reference.trace)),
                 "homeostasis": jnp.asarray(np.zeros_like(reference.homeostasis)),
                 "connections": jnp.asarray(initial["connections"]),
                 "previous": jnp.asarray(np.zeros_like(reference.previous)),
                 "forecast": jnp.asarray(np.zeros_like(reference.forecast)),
                 "plasticity": jnp.asarray(.04, jnp.float32)}
        # Recreate the reference from identical initial state before comparing.
        reference = DevelopmentalSheet(sheet=4, channels=7, seed=11, plasticity=.04)
        expected = reference.observe(frame, action)
        actual_state, actual = developmental_jax.step(state, jnp.asarray(frame), action)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(np.asarray(actual_state["connections"]), reference.connections,
                                   rtol=2e-5, atol=2e-6)

    def test_smoke_resumes_completed_conditions(self):
        config = "experiments/akash-developmental-smoke.json"
        with tempfile.TemporaryDirectory() as folder:
            first = evidence.run(config, folder, 4, require_gpu=False)
            second = evidence.run(config, folder, 4, require_gpu=False)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "completed")
            self.assertEqual(len(second["conditions"]), 4)
            self.assertTrue((Path(folder) / "conditions/coherent-s11/checkpoint-visual.npz").exists())

    def test_injected_failure_recovers_completed_condition(self):
        config = "experiments/akash-developmental-smoke.json"
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict("os.environ", {"DEVELOPMENTAL_INJECT_FAILURE_AFTER": "1"}):
            with self.assertRaises(RuntimeError):
                evidence.run(config, folder, 4, require_gpu=False)
            self.assertEqual(len(json.loads(
                (Path(folder) / "evidence.json").read_text())["conditions"]), 1)
        report = evidence.run(config, folder, 4, require_gpu=False)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["conditions"]), 4)


if __name__ == "__main__":
    unittest.main()
