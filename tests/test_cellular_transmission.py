import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from experiments.cellular_transmission import batch, run
from src import cellular_signal as model


class CellularTransmissionTests(unittest.TestCase):
    def test_target_is_not_sign_label(self):
        pulse,target,sign,distance=batch(112,16)
        self.assertEqual(pulse.shape,(16,16,16,1))
        np.testing.assert_allclose(np.abs(pulse).sum(axis=(1,2,3)),1)
        np.testing.assert_allclose(target.sum(axis=(1,2,3)),1)
        self.assertTrue(all(2<=d<=4 for d in distance))
        self.assertTrue(set(sign).issubset({-1.,1.}))

    def test_no_messages_blocks_nonlocal_signed_signal(self):
        pulse,target,_,_=batch(156,1)
        params=model.init(jax.random.PRNGKey(2),channels=4)
        response1=model.target_signal(params,jnp.asarray(pulse),
                                      jnp.asarray(target),ticks=7,
                                      allow_messages=False)
        response2=model.target_signal(params,jnp.asarray(-pulse),
                                      jnp.asarray(target),ticks=7,
                                      allow_messages=False)
        np.testing.assert_allclose(np.asarray(response1),np.asarray(response2),
                                   rtol=0,atol=1e-6)

    def test_output_shape(self):
        pulse,target,_,_=batch(6,2)
        params=model.init(jax.random.PRNGKey(1),channels=4)
        result=model.rollout(params,jnp.asarray(pulse),jnp.asarray(target),5)
        self.assertEqual(result.shape,(5,2,16,16,1))

    def test_one_step_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run(out=folder,seeds=(3,),steps=1,size=12,ticks=7,
                       batch_size=2,channels=4)
            self.assertEqual(len(result["results"]),1)
            self.assertTrue((Path(folder)/"metrics.json").exists())


if __name__=="__main__":
    unittest.main()
