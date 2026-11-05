"""Dataset leakage, axis order, hash integrity and oracle scoring."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from experiments.external_video_bridge import export, verify, evaluate


class ExternalBridgeTests(unittest.TestCase):
    def test_export_is_pairwise_and_no_truth_inside_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"export"
            m=export(out,train_pairs=2,validation_pairs=2,test_pairs=2)
            self.assertTrue(verify(out))
            for split in ("train","validation","test"):
                with np.load(out/f"{split}.npz") as data:
                    self.assertEqual(set(data.files),{"inputs","targets"})
                    x,y=data["inputs"],data["targets"]
                    self.assertEqual(x.shape,(4,8,3,32,32))
                    self.assertEqual(y.shape,(4,10,3,32,32))
                    np.testing.assert_array_equal(x[::2,-1],x[1::2,-1])
                    self.assertGreater(np.max(np.abs(y[::2]-y[1::2])),0.)
            self.assertEqual(m["splits"]["test"]["independent_scene_pairs"],2)
            with self.assertRaises(FileExistsError):
                export(out,train_pairs=2,validation_pairs=2,test_pairs=2)

    def test_perfect_oracle_score_and_modified_dataset_fail(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"dataset"
            export(out,train_pairs=2,validation_pairs=2,test_pairs=2)
            with np.load(out/"test.npz") as data:
                prediction=data["targets"].copy()
            pred=Path(td)/"prediction.npz"
            np.savez_compressed(pred,predictions=prediction)
            report=evaluate(out,pred)
            self.assertAlmostEqual(report["rgb_mse"],0.)
            self.assertAlmostEqual(report["success_rate"],1.)
            self.assertEqual(len(report["pair_success"]),2)
            with (out/"test.npz").open("ab") as f:
                f.write(b"X")
            with self.assertRaisesRegex(ValueError,"modified"):
                verify(out)

    def test_prediction_shape_and_numeric_range_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"dataset"
            export(out,train_pairs=1,validation_pairs=1,test_pairs=1)
            with np.load(out/"test.npz") as data:
                x=data["targets"].copy()
            invalid=Path(td)/"pred.npz"
            np.savez_compressed(invalid,predictions=x.transpose(0,1,3,4,2))
            with self.assertRaisesRegex(ValueError,"dimensions"):
                evaluate(out,invalid)
            x[0,0,0,0,0]=np.nan
            np.savez_compressed(invalid,predictions=x)
            with self.assertRaisesRegex(ValueError,"Nonfinite"):
                evaluate(out,invalid)


if __name__ == "__main__":
    unittest.main()
