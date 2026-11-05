import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from experiments import roadmap_44_live_mirror as m
from experiments import frozen_visual_bridge as visual


class IntegratedMirrorTests(unittest.TestCase):
    def test_checkpoint_selection_hash(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/"cnn.npz"
            weights,_=visual.checkpoint("cnn",smoke=True,width=2)
            np.savez(file,**{k:np.asarray(v) for k,v in weights.items()})
            manifest=Path(d)/"selection.json"
            manifest.write_text(json.dumps({"selected":{
                "name":"cnn","checkpoint_sha256":"wrong"}}))
            with self.assertRaises(ValueError):m.selected(manifest,file)

    def test_synchronized_offline_harness_not_live_claim(self):
        frames=np.zeros((9,16,16,3),np.float32)
        frames[:,4:6,4:6]=.3
        p,_=visual.checkpoint("cnn",smoke=True,width=2)
        result=m.replay_stream(frames,np.arange(9,dtype=np.int64)*100000000,
                               "cnn",p,context=3,lead=2)
        self.assertEqual(result["selected"]["scored"],5)
        self.assertEqual(result["persistence"]["scored"],5)
        self.assertGreater(result["selected"]["median_observed_horizon_ms"],0)


if __name__=="__main__":
    unittest.main()
