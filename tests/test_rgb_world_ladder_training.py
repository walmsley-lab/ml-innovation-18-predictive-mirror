import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from experiments.rgb_world_ladder_training import run,specification,objective


class WorldLadderTrainingTests(unittest.TestCase):
    def test_models_have_common_pixel_objective(self):
        batch=jnp.zeros((1,5,20,20,3),jnp.float32)
        for name in ("cnn","transport","nca","routing","hierarchical"):
            params,predict=specification(name,jax.random.PRNGKey(1),4,2,1)
            loss=objective(params,batch,predict,3,2)
            self.assertTrue(np.isfinite(float(loss)))

    def test_one_step_two_curricula(self):
        with tempfile.TemporaryDirectory() as directory:
            result=run(directory,models=("cnn",),seeds=(1,),
                       curricula=("linear","mixed"),
                       steps=1,context=3,horizon=2,size=20,
                       width=4,ticks=1,batch_size=2,
                       train_per_domain=2,val_per_domain=1,
                       test_per_domain=1)
            self.assertEqual(len(result["results"]),2)
            for run_data in result["results"]:
                self.assertEqual(run_data["selected_step"] in (0,1),True)
                self.assertTrue((Path(directory)/
                    f"cnn-seed1-{run_data['curriculum']}-best.npz").exists())
                self.assertEqual(len(run_data["test"]),7)


if __name__=="__main__":
    unittest.main()
