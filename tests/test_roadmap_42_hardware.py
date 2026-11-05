import unittest
import numpy as np
from experiments import roadmap_42_hardware as m


class HardwareAuditTests(unittest.TestCase):
    def test_physical_peripheral_counts_not_invented_energy(self):
        a=m.accounting(16,4,3,1,1)
        b=m.accounting(32,4,3,1,1)
        self.assertEqual(b["cell_updates"],4*a["cell_updates"])
        self.assertIsNone(b["physical_energy_joules"])
        self.assertLess(a["ideal_packed_4bit_state_bytes_lower_bound"],
                        a["numpy_state_bytes_float32"])
        self.assertEqual(a["rgb_output_adc_transactions_lower_bound"],16*16*3*3)

    def test_metric_and_latency(self):
        report=m.run(None,sizes=(16,),modes=("full","quantized_4bit"),
                     train_count=2,test_count=2,context=3,horizon=2,
                     ticks=1,repeats=2)
        self.assertEqual(len(report["rows"]),2)
        for row in report["rows"]:
            self.assertIn("constant_velocity",row["baselines"])
            self.assertIn("2",row["scores"])
            self.assertTrue(np.isfinite(row["cpu_latency"]["median_ms"]))
            self.assertEqual(row["hardware"]["actual_state_dtype"],
                             "float32 even in quantized_4bit mode")


if __name__=="__main__":
    unittest.main()
