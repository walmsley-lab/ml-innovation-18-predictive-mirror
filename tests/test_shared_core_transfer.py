import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src import shared_cellular_core as model
from experiments.shared_core_transfer import run


class SharedCoreTests(unittest.TestCase):
    def test_both_domains_use_exact_same_core(self):
        p=model.init(jax.random.PRNGKey(3),4)
        self.assertEqual(p["core"].shape,(3,3,4,4))
        pulse=jnp.zeros((2,16,16,1)).at[:,8,6,0].set(1)
        dest=jnp.zeros_like(pulse).at[:,8,8,0].set(1)
        self.assertEqual(model.signal(p,pulse,dest,4).shape,(2,4))
        frames=jnp.zeros((2,3,16,16,3))
        self.assertEqual(model.rgb(p,frames,2).shape,(2,2,16,16,3))

    def test_short_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=run(tmp,seeds=(2,),pre_steps=1,rgb_steps=1,
                       size=16,context=3,horizon=2,channels=4)
            self.assertEqual(set(report["results"][0]["arms"]),
                             {"scratch","pretrained","shifted_pretrain"})
            self.assertEqual(report["pretrain_steps"],1)


if __name__=="__main__":
    unittest.main()
