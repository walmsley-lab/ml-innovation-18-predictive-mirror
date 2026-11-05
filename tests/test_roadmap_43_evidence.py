import json
import tempfile
from pathlib import Path
import unittest
from experiments import roadmap_43_evidence as m


def phase_report():
    return {"roadmap_id":37,"results":[{"seed":s,"mode":arm,
        "heldout":{"chirp":{"1":{"mse":loss,"motion_mse":loss+.1}}}}
        for s in (11,23,41) for arm,loss in (("plain",.3),("coupled",.2))]}


class EvidenceComparisonTests(unittest.TestCase):
    def test_paired_same_protocol_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"metrics.json"
            source.write_text(json.dumps(phase_report()))
            r=m.summarize([source])
            group=next(x for x in r["groups"] if x["metric"]=="mse")
            pair=group["paired"]["coupled minus plain"]
            self.assertEqual(pair["n_paired_seeds"],3)
            self.assertLess(pair["mean_difference"],0)
            self.assertEqual(len(pair["bootstrap_ci_95"]),2)

    def test_cannot_pool_privileged_structure_and_rgb(self):
        skeleton={"roadmap_id":38,"rows":[{"seed":11,"arm":"graph",
                   "scores":{"learned":{"1":{"visible_joint_mae_px":2.}}}}]}
        particle={"roadmap_id":39,"rows":[{"seed":11,"input":"rgb_centroids",
                   "messages":True,"scores":{"particle_nca":{
                     "1":{"position_mae_px":3.}}}}]}
        with tempfile.TemporaryDirectory() as directory:
            files=[]
            for i,data in enumerate((phase_report(),skeleton,particle)):
                path=Path(directory)/f"{i}.json"
                path.write_text(json.dumps(data));files.append(path)
            groups=m.summarize(files)["groups"]
            representations=set(g["representation"] for g in groups)
            self.assertEqual(representations,{
                "observed_rgb","given_skeleton_joints","rgb_centroids"})
            self.assertFalse(any(g["paired"] for g in groups
                                 if g["representation"]!="observed_rgb"))

    def test_latent_and_hardware_reports_remain_descriptive(self):
        latent={"roadmap_id":36,"cases":{
            "linear":{"latent":{"1":{"mse":.12,"motion_mse":.2}},
                      "persistence":{"1":{"mse":.15,"motion_mse":.25}}}}}
        hardware={"roadmap_id":42,"rows":[
            {"size":16,"mode":mode,"scores":{"1":{"mse":score}},
             "baselines":{"persistence":{"1":{"mse":.18}}}}
            for mode,score in (("full",.21),("quantized_4bit",.20))]}
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i,report in enumerate((latent,hardware)):
                path=Path(directory)/f"report{i}.json"
                path.write_text(json.dumps(report))
                paths.append(path)
            result=m.summarize(paths)
            self.assertEqual(len(result["sources"]),2)
            latent_groups=[x for x in result["groups"] if x["roadmap_id"]==36]
            self.assertTrue(latent_groups)
            self.assertTrue(all(x["status"]=="descriptive_only"
                                for x in result["groups"]))
            self.assertIn("latent",latent_groups[0]["arms"])
            self.assertNotEqual(latent_groups[0]["scope"],"circuit-hardware")
            hardware_groups=[x for x in result["groups"] if x["roadmap_id"]==42]
            self.assertEqual(len(hardware_groups),1)
            self.assertEqual(set(hardware_groups[0]["arms"]),
                             {"full","quantized_4bit","persistence"})

    def test_frozen_visual_bridge_inputs_and_control_deduplication(self):
        sample={"protocol":"frozen-visual-bridge-v1",
                "cases":{"linear/none":{
                    arm:{"metrics":{
                        arm:{"1":{"mse":val,"motion_mse":val+.1}},
                        "persistence":{"1":{"mse":.3}},
                        "constant_velocity":{"1":{"mse":.2}}}}
                    for arm,val in (("nca",.25),("cnn",.22),("transport",.24))}}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"frozen.json"
            path.write_text(json.dumps(sample))
            rows=m._rows(sample)
            self.assertEqual(sum(r["model"]=="persistence" for r in rows),1)
            self.assertEqual(sum(r["model"]=="cnn" for r in rows),2)
            result=m.summarize([path])
            self.assertEqual(result["sources"][0]["roadmap_id"],
                             "frozen-visual-bridge-v1")
            self.assertTrue(all(g["status"]=="descriptive_only"
                                for g in result["groups"]))

    def test_duplicate_cannot_impersonate_replicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"r.json"
            report=phase_report()
            report["results"].append(report["results"][0].copy())
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                m.summarize([path])


if __name__=="__main__":
    unittest.main()
