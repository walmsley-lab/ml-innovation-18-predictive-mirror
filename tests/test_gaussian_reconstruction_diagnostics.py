import tempfile
import unittest
from pathlib import Path
import jax
import numpy as np
from src import world3d, rgb_gaussian_nca,gaussian_slot_matching
from experiments.gaussian_reconstruction_diagnostics import run,evaluate


class ReconstructionTests(unittest.TestCase):
    def test_permutation_matching_and_missing(self):
        matches,missing=gaussian_slot_matching.assignment(
            np.asarray([[8.,0.],[0.,8.]]))
        self.assertEqual({(a,b) for a,b,_ in matches},{(0,1),(1,0)})
        self.assertEqual(missing,[])
        _,missing=gaussian_slot_matching.assignment(np.asarray([[100.]]),40.)
        self.assertEqual(missing,[0])

    def test_same_renderer_and_observation_only(self):
        record=world3d.episode(21,6,16)
        params=rgb_gaussian_nca.init(jax.random.PRNGKey(1),3,4)
        report=evaluate(record,params,5)
        self.assertEqual(report["inferred_rgb"].shape,(16,16,3))
        self.assertGreaterEqual(report["oracle"]["rgb_mse"],0)
        self.assertTrue(np.isfinite(report["extra_encoder_error_mse"]))
        self.assertEqual(report["slot_count"],3)

    def test_explicit_smoke_and_png_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run(directory)
            report=run(directory,untrained_smoke=True,seeds=(1,),
                       context=3,size=16)
            image=Path(directory)/report["rows"][0]["gallery"]
            self.assertEqual(image.read_bytes()[:8],b"\x89PNG\r\n\x1a\n")


if __name__=="__main__":
    unittest.main()
