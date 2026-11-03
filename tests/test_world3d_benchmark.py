import tempfile
import unittest
from pathlib import Path
import numpy as np
from src import world3d,world3d_benchmark as bench
from experiments.world3d_benchmark import run


class World3DBenchmarkTests(unittest.TestCase):
    def test_reprojection_identity_and_truth_is_privileged(self):
        record=world3d.episode(4,steps=5,size=20)
        rgb=record["rgb"][0]
        result=bench.reproject(rgb,record["depth"][0],record["K"],
                               record["T_world_camera"][0],
                               record["T_world_camera"][0])
        mask=record["instance_id"][0]>=0
        np.testing.assert_allclose(result[mask],rgb[mask],atol=1e-6)
        self.assertEqual(set(world3d.model_inputs(record)),{"rgb"})

    def test_scores_and_frozen_runner(self):
        rec=[world3d.episode(5,steps=6,size=16)]
        perfect=rec[0]["rgb"][3:6][None]
        scores=bench.score(rec,perfect,context=3,leads=(1,3))
        self.assertAlmostEqual(scores["3"]["mse"],0)
        with tempfile.TemporaryDirectory() as folder:
            report=run(Path(folder)/"results.json",episodes=2,context=3,
                       horizon=2,size=16)
            self.assertTrue(report["controls"]["oracle_depth_reprojection"]["privileged"])


if __name__=="__main__":
    unittest.main()
