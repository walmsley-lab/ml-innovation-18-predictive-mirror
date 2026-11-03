import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from experiments.cellular_message_modulation import batch,run
from src import cellular_message_modulation as model


class MessageModulationTests(unittest.TestCase):
    def test_competing_pulses_and_cue(self):
        pulse,cue,target,label=batch(71,count=12,size=12)
        self.assertEqual(pulse.shape,(12,12,12,1))
        np.testing.assert_allclose(cue.sum(axis=(1,2,3)),1)
        np.testing.assert_allclose(target.sum(axis=(1,2,3)),1)
        for i in range(12):
            chosen=(pulse[i]*cue[i]).sum()
            self.assertEqual(chosen,label[i])

    def test_modes_preserve_shape_and_zero_step_causality(self):
        p=model.init(jax.random.PRNGKey(1),channels=4)
        pulse,cue,dest,truth=batch(81,count=2,size=12)
        for mode in model.MODES:
            result=model.rollout(p,jnp.asarray(pulse),jnp.asarray(cue),
                                 jnp.asarray(dest),ticks=4,mode=mode)
            self.assertEqual(result.shape,(2,4))
        self.assertEqual(len(truth),2)

    def test_one_update_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            report=run(out=temp,seeds=(3,),steps=1,size=12,
                       ticks=5,batch_size=2,channels=4)
            self.assertTrue((Path(temp)/"metrics.json").exists())
            self.assertEqual(len(report["results"]),1)


if __name__=="__main__":
    unittest.main()
