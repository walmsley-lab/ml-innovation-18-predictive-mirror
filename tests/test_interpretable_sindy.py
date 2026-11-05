import tempfile
import unittest
import numpy as np
from src import interpretable_sindy as s
from experiments.interpretable_sindy import run


class SINDyTests(unittest.TestCase):
    def test_known_constant_velocity_system(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(500,4))
        target=x.copy();target[:,:2]+=x[:,2:]
        for method in ("sindy","dmd","affine"):
            model=s.fit(x,target,method,threshold=.02)
            pred=s.next_step(model,x)
            self.assertLess(np.sqrt(np.mean((pred-target)**2)),.015)
            self.assertEqual(len(s.equations(model)),4)

    def test_state_library_and_invalid(self):
        library,names=s.library(np.ones((4,4)),degree=2)
        self.assertEqual(library.shape,(4,15))
        self.assertEqual(len(names),15)
        with self.assertRaises(ValueError):
            s.fit(np.ones((3,3)),np.ones((3,3)))

    def test_ood_run_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            r=run(folder+"/metrics.json",seeds=(1,),
                  noise_levels=(0,),test_domains=("linear",),horizon=2)
            self.assertEqual(len(r["results"]),3)


if __name__=="__main__":
    unittest.main()
