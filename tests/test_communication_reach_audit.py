import tempfile
import unittest
import numpy as np
from src import communication_reach as reach
from experiments.communication_reach_audit import run


class ReachAuditTests(unittest.TestCase):
    def test_radial_known_impulse(self):
        f=np.zeros((11,11))
        f[5,5]=1;f[5,8]=.5
        profile=reach.radial(f,(5,5))
        self.assertEqual(profile["measured_reach_px"],3.)
        self.assertGreater(profile["response_by_radius"][3],0)

    def test_isolated_cell_and_flow_shapes(self):
        result=reach.compare(1,size=12,context=3,horizon=2,ticks=1,channels=3)
        self.assertEqual(len(result["paths"]["nca_rgb"]["per_lead"]),2)
        self.assertEqual(len(result["paths"]["cellular_hidden"]["per_tick"]),1)
        self.assertIn("flow_response",result["paths"]["learned_flow"])

    def test_memory_probe_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            report=run(directory+"/metrics.json",seeds=(1,),
                       size=12,context=3,horizon=1,ticks=1)
            self.assertEqual(len(report["memory"]["rows"]),3)


if __name__=="__main__":
    unittest.main()
