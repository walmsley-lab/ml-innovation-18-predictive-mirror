import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import cellular_references as model
from src.cellular_routing_probe import analyze
from experiments.cellular_references import sample
from experiments.cellular_routing_probe import run


class RoutingProbeTests(unittest.TestCase):
    def test_causal_lesions_and_normalization(self):
        examples=sample(12,3,size=8,low=2,high=4)
        params=model.init(jax.random.PRNGKey(1),channels=4)
        score=analyze(params,examples,ticks=2)
        self.assertEqual(set(score),{"uniform","learned","shuffled","key_lesion"})
        self.assertAlmostEqual(score["uniform"]["source_routing_mass"],1/64)
        self.assertEqual(len(score["key_lesion"]["per_episode_signed_output"]),3)

    def test_frozen_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/"weights.npz"
            p=model.init(jax.random.PRNGKey(1),channels=4)
            np.savez_compressed(f,**jax.tree.map(np.asarray,p))
            report=run(f,out=Path(folder)/"metrics.json",episodes=2,size=8)
            self.assertEqual(report["episodes"],2)


if __name__ == "__main__":
    unittest.main()
