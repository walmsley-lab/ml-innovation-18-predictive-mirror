import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from experiments.cellular_references import run,sample
from src import cellular_references as model


class CellularReferencesTests(unittest.TestCase):
    def test_sample_has_unique_source_and_target(self):
        pulse,target,sign=sample(9,12,size=10)
        np.testing.assert_allclose(np.abs(pulse).sum(axis=(1,2,3)),1)
        np.testing.assert_allclose(target.sum(axis=(1,2,3)),1)
        self.assertTrue(set(sign).issubset({-1.,1.}))

    def test_routing_normalizes_and_mode_changes(self):
        p=model.init(jax.random.PRNGKey(1),channels=4)
        pulse,target,_=sample(10,2,size=8)
        a,weights=model.rollout(p,jnp.asarray(pulse),jnp.asarray(target),2,
                                "learned",True)
        self.assertEqual(a.shape,(2,))
        self.assertEqual(weights.shape,(2,64,64))
        np.testing.assert_allclose(np.asarray(weights.sum(axis=-1)),1,atol=1e-5)
        b=model.rollout(p,jnp.asarray(pulse),jnp.asarray(target),2,"uniform")
        self.assertEqual(b.shape,a.shape)

    def test_one_step_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            result=run(out=temp,seeds=(3,),steps=1,size=8,ticks=2,channels=4)
            self.assertEqual(len(result["results"]),1)
            self.assertTrue((Path(temp)/"metrics.json").exists())


if __name__=="__main__":
    unittest.main()
