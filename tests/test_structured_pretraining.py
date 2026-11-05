import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from src import structured_pretraining as pre
from experiments.structured_pretraining_transfer import run


class StructuredPretrainingTests(unittest.TestCase):
    def test_all_sources_shapes(self):
        for mode in pre.SOURCES:
            result=pre.samples(1,2,mode,context=3,horizon=2,size=20)
            self.assertEqual(result.shape,(2,5,20,20,3))
            self.assertTrue(np.isfinite(result).all())

    def test_identical_param_dict_across_transfer_arms(self):
        a,predict=pre.architecture("cnn",jax.random.PRNGKey(1),4,1,2)
        self.assertEqual(set(a),{"conv1","conv2","readout","bias"})
        images=jnp.zeros((2,3,20,20,3))
        self.assertEqual(predict(a,images).shape,(2,2,20,20,3))

    def test_one_update(self):
        with tempfile.TemporaryDirectory() as directory:
            report=run(directory,architectures=("cnn",),
                       sources=("wave","shuffled","scratch"),seeds=(1,),
                       pretrain_steps=1,finetune_steps=1,
                       context=3,horizon=2,size=20,width=4,batch_size=2)
            self.assertEqual(len(report["results"]),3)
            self.assertEqual(set(x["total_updates"] for x in report["results"]),{2})


if __name__=="__main__":
    unittest.main()
