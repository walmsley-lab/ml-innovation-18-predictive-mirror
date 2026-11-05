import tempfile
import unittest
import numpy as np
from experiments import roadmap_41_transfer as m


class SignalVisualTransferTests(unittest.TestCase):
    def test_rgb_world_split_shapes(self):
        a=m.rgb_pool(31000,2,3,2,16,{"balls":1,"occlusion":True})
        b=m.rgb_pool(91000,2,3,2,16,{"balls":1,"occlusion":True})
        self.assertEqual(a.shape,(2,5,16,16,3))
        self.assertFalse(np.array_equal(a,b))

    def test_equal_total_update_accounting(self):
        with tempfile.TemporaryDirectory() as p:
            r=m.run(p,seeds=(3,),pre_steps=1,target_steps=1,
                    context=3,horizon=2,channels=2)
            self.assertEqual(len(r["results"]),4)
            rows={x["arm"]:x for x in r["results"]}
            self.assertEqual(rows["signal_pretrained"]["total_updates"],2)
            self.assertEqual(rows["scratch_equal_total_updates"]["total_updates"],2)
            self.assertEqual(rows["scratch_finetune_only"]["total_updates"],1)
            self.assertEqual(rows["shuffled_signal_labels"]["target_rgb_updates"],1)
            self.assertIn("camera_motion",rows["signal_pretrained"]["heldout"])


if __name__=="__main__":
    unittest.main()
