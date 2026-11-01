"""Training families, disjoint generator identities, and tiny CPU sanity."""
import tempfile
import unittest
import numpy as np
from experiments import rgb_curriculum as exp


class CurriculumTests(unittest.TestCase):
    def test_reproducible_families_and_shapes(self):
        a = exp.batch(31000, 2, 4, 3, 20, 0)
        b = exp.batch(31000, 2, 4, 3, 20, 0)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape, (2,7,20,20,3))
        self.assertEqual(len(exp.SCENARIOS),4)
        self.assertGreater(np.max(np.abs(a-exp.batch(31000,2,4,3,20,1))),0)

    def test_one_step_writes_paired_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            result=exp.run(output=folder,seeds=(3,),steps=1,batch_size=1,
                           context=3,horizon=2,ticks=1,size=16,channels=4)
            self.assertEqual(len(result["runs"]),1)
            self.assertIn("curriculum",result["summary"])
            self.assertIn("single",result["summary"])
            self.assertIn("constant_velocity",result["controls"])
