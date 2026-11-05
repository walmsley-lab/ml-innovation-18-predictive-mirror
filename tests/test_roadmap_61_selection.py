import json
import tempfile
from pathlib import Path
import unittest
from experiments import roadmap_61_selection as m


def candidate(name,mse,latency):
    return {"name":name,"checkpoint_sha256":name+"123","validation_mse":mse,
            "latency_ms":latency,"device":"CPU-A","protocol":"rgb-1s",
            "validation_session_sha256":"val-session-1","evidence":"trained_checkpoint"}


class ModelSelectionTests(unittest.TestCase):
    def test_latency_gate_and_validation_only(self):
        c=[candidate("nca",.1,150),candidate("cnn",.2,8),
           candidate("transport",.16,35)]
        self.assertEqual(m.select(c,100)["name"],"transport")
        self.assertIsNone(m.select(c,4))
        with self.assertRaises(ValueError):
            m.select([{**c[0],"evidence":"untrained_smoke"}],200)
        with self.assertRaises(ValueError):
            m.select([{**c[0],"latency_ms":0}],200)
        with self.assertRaises(ValueError):
            m.select([c[0],{**c[1],"device":"GPU"}],200)

    def test_test_scores_cannot_change_selection(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"val.json"
            path.write_text(json.dumps({"candidates":[candidate("cnn",.2,7),
                                                    candidate("nca",.15,11)]}))
            before=m.run(path,out=None)
            test=Path(d)/"test.json"
            test.write_text(json.dumps({"checkpoint_sha256":"nca123",
                                       "test_session_sha256":"test-2","mse":.9}))
            after=m.run(path,test,out=None)
            self.assertEqual(before["selected"],after["selected"])
            self.assertEqual(after["test"]["selected_model_mse"],.9)


if __name__=="__main__":
    unittest.main()
