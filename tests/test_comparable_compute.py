import tempfile
import unittest

from experiments.comparable_compute import distribution, run


class ComparableComputeTests(unittest.TestCase):
    def test_distribution_with_variance(self):
        result=distribution([.001,.002,.003,.004])
        self.assertEqual(result["n"],4)
        self.assertEqual(result["median_ms"],2.5)
        self.assertGreater(result["p95_ms"],result["p05_ms"])
        self.assertGreater(result["std_ms"],0)

    def test_cpu_smoke_separates_compile_and_rollout(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run(folder+"/report.json",models=("cnn",),seeds=(1,),
                       size=20,context=3,horizon=2,batch_size=2,
                       width=4,ticks=1,train_steps=1,
                       warmup=1,repetitions=2)
            row=result["results"][0]
            self.assertGreaterEqual(row["compile_ms"]["inference_rollout"],0)
            self.assertEqual(row["inference_rollout_ms"]["n"],2)
            self.assertEqual(row["training_steady_step_ms"]["n"],1)
            self.assertEqual(row["workload"]["horizon"],2)
            self.assertIn("jax",result["environment"])


if __name__=="__main__":
    unittest.main()
