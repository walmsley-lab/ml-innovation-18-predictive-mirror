"""Offline tests for Akash GPU campaign plumbing. No network or GPU required."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deploy import gpu_campaign as campaign
from deploy import gpu_worker as worker

IMAGE = "ghcr.io/walmsley-lab/ml-innovation-18-predictive-mirror@sha256:" + "a"*64


class Tests(unittest.TestCase):
    def test_inventory_chunks_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"sample.npz").write_bytes(b"weights")
            item=worker.inventory(root)["files"][0]
            self.assertEqual(item["sha256"],hashlib.sha256(b"weights").hexdigest())
            self.assertEqual(base64.b64decode(worker.read_chunk(root,"sample.npz",0,20)),b"weights")
            with self.assertRaises(ValueError):worker.read_chunk(root,"../bad",0,20)

    def test_worker_keeps_status_outside_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=root/"cfg.json";cfg.write_text("{}")
            with patch.object(worker,"gpu_available",return_value=True),\
                 patch.object(worker.subprocess,"run"),\
                 patch.object(worker.time,"monotonic",side_effect=[0,61]),\
                 patch.object(worker.time,"sleep"):
                ok=worker.run_campaign(cfg,root/"campaign",1,worker.hash_file(cfg),60)
            self.assertTrue(ok)
            self.assertEqual(json.loads((root/"gpu_status.json").read_text())["status"],"completed")
            self.assertFalse((root/"campaign/gpu_status.json").exists())

    def test_bad_image_config_fails_fast_with_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=root/"cfg.json";cfg.write_text("{}")
            with patch.object(worker,"gpu_available") as gpu, \
                 patch.object(worker.time,"monotonic",side_effect=[0,61]), \
                 patch.object(worker.time,"sleep"):
                ok=worker.run_campaign(cfg,root/"campaign",1,"0"*64,60)
            self.assertFalse(ok)
            gpu.assert_not_called()
            self.assertEqual(json.loads((root/"gpu_status.json").read_text())["status"],"failed")

    def test_sdl_and_manifest(self):
        template=Path("deploy/akash-gpu.yaml").read_text()
        output=campaign.render_sdl(template,IMAGE,"experiments/gpu-smoke.json","b"*64,1,60)
        for text in ("RUN_MODE=campaign","CAMPAIGN_APPROVED=1","ARTIFACT_HTTP=0"):
            self.assertIn(text,output)
        count,sha=campaign.load_manifest("experiments/gpu-smoke.json")
        self.assertEqual((count,len(sha)),(1,64))
        with self.assertRaises(ValueError):
            campaign.render_sdl(template,IMAGE.replace("@sha256:",":tag-"),
                                "experiments/gpu-smoke.json","b"*64,1,60)

    def test_dseq_and_close_state(self):
        self.assertEqual(campaign.dseq_from_event({"outputs":{"dseq":"1234"}}),"1234")
        self.assertIsNone(campaign.dseq_from_event({"outputs":{"dseq":"no"}}))
        self.assertTrue(campaign.is_closed({"deployment":{"state":"closed"}}))

    def test_download_private(self):
        data=b"private-checkpoint"
        item={"path":"pilot/model.npz","bytes":len(data),
              "sha256":hashlib.sha256(data).hexdigest()}
        def fake(dseq,command,context):
            if command[0]=="inventory":return {"files":[item]}
            return {"data":base64.b64encode(data).decode()}
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,"remote",side_effect=fake):
            self.assertEqual(campaign.download({"dseq":"1","out":tmp},None,1024),1)
            self.assertEqual((Path(tmp)/"artifacts/pilot/model.npz").read_bytes(),data)
            with self.assertRaises(ValueError):
                campaign.download({"dseq":"1","out":tmp},None,1)

    def test_plan_does_not_spend(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,"cli",return_value="ok") as command:
            campaign.main(["plan","--image",IMAGE,"--out",str(Path(tmp)/"plan")])
            self.assertEqual(command.call_count,2)
            self.assertIn("--dry-run",command.call_args_list[1].args[0])
            with self.assertRaises(SystemExit):
                campaign.main(["run","--image",IMAGE,"--out",str(Path(tmp)/"run")])
            self.assertFalse((Path(tmp)/"run/state.json").exists())


if __name__=="__main__":
    unittest.main()
