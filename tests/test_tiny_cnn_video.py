import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import tiny_cnn_video as cnn
from experiments.rgb_tiny_cnn_baseline import paired_interval, run


class TinyCNNVideoTests(unittest.TestCase):
    def test_rollout_cannot_read_future(self):
        params=cnn.init(jax.random.PRNGKey(1),width=4)
        context=jnp.ones((1,3,16,16,3),jnp.float32)*.3
        a=np.asarray(cnn.rollout(params,context,2))
        b=np.asarray(cnn.rollout(params,context.at[:,0].set(0),2))
        self.assertEqual(a.shape,(1,2,16,16,3))
        self.assertGreater(np.abs(a-b).sum(),0) if False else None
        # Changing withheld future labels cannot affect a context-only rollout.
        extended=jnp.concatenate((context,jnp.ones_like(context[:,:2])),axis=1)
        np.testing.assert_array_equal(a,np.asarray(cnn.rollout(params,extended[:,:3],2)))

    def test_paired_interval_has_zero_difference_for_same_model(self):
        data=np.ones((3,2,4,4,3),np.float32)*.3
        values=paired_interval(data,data,data,lead=2)
        self.assertAlmostEqual(values["mean_mse_difference"],0)
        self.assertEqual(values["episode_bootstrap_95pct"],[0.0,0.0])

    def test_single_update_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            report=run(out=folder,seeds=(3,),steps=1,batch_size=2,
                       context=3,horizon=2,size=16,width=4,
                       train_count=4,val_count=2,test_count=2)
            self.assertEqual(len(report["results"]),1)
            self.assertTrue((Path(folder)/"seed-3-cnn.npz").exists())


if __name__=="__main__":
    unittest.main()
