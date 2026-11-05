"""70: evidence ledger for connectome and neural-response experiments.

A passing synthetic campaign NEVER upgrades a biological hypothesis to
measured support. Each part retains input-source type and model limitations;
neural agreement has no measured support unless independent data are loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
from experiments import (
    roadmap_62_connectome_contract as e62,
    roadmap_63_retinotopy as e63,
    roadmap_64_graph_dynamics as e64,
    roadmap_65_lesions as e65,
    roadmap_66_neural_agreement as e66,
    roadmap_67_harmonics as e67,
    roadmap_68_substrates as e68,
    roadmap_69_topology_transfer as e69)

MODULES={62:e62,63:e63,64:e64,65:e65,66:e66,
         67:e67,68:e68,69:e69}
CRITERIA={
    "connectome_predictive_prior":"requires measured connectivity, fair "
       "non-connectomic model, independent test and held-out accuracy",
    "lesion_causal_validation":"requires independent biological perturbation "
       "and predicted response before the actual lesion experiment",
    "neural_response_agreement":"requires measured stimulus/neural records, "
       "independent sessions, noise ceiling and permutation controls",
    "cross_substrate_efficiency":"requires measured cost on actual devices "
       "including peripheral inputs/outputs, not graph-edge counts"}


def assert_evidence(items):
    for item in items:
        if item["source_type"] not in ("synthetic","measured"):
            raise ValueError("Unknown biological evidence level")
        if item["source_type"]=="measured" and not item.get("provenance"):
            raise ValueError("Measured result without provenance cannot enter ledger")


def run(out="runs/roadmap-70",seeds=(11,23)):
    if len(seeds)<2 or len(set(seeds))!=len(seeds):
        raise ValueError("At least two distinct seeds needed")
    a,ids=e62.toy()
    sections={
        62:{"roadmap_id":62,**e62.summarize(
            a,{"source_id":"toy","source_type":"synthetic"})},
        63:e63.run(out=None,seeds=seeds),
        64:e64.run(out=None,seeds=seeds),
        65:e65.run(out=None,seeds=seeds),
        66:e66.run(out=None),
        67:e67.run(out=None,seeds=seeds),
        68:e68.run(out=None,seeds=seeds),
        69:e69.run(out=None,seeds=seeds)}
    parts=[]
    for number,module in MODULES.items():
        result=sections[number]
        label=result["source_type"]
        parts.append({"roadmap_id":number,"source_type":label,
                     "module_sha256":hashlib.sha256(
                         Path(module.__file__).read_bytes()).hexdigest(),
                     "evidence_scope":("synthetic_control_only" if label=="synthetic"
                                       else "external_record_requires_review"),
                     "provenance":result.get("source_id"),
                     "results":result})
    assert_evidence(parts)
    measured=any(p["source_type"]=="measured" for p in parts)
    findings={name:{"status":"not_demonstrated",
                   "acceptance_requirement":rule}
              for name,rule in CRITERIA.items()}
    manifest={"roadmap_id":70,"protocol":"biological-branch-v1",
              "seeds":list(map(int,seeds)),"parts":parts,
              "evidence_status":("external_data_requires_independent_review" if measured
                                  else "synthetic_controls_only"),
              "claims":findings,
              "limitations":"No measured connectome/neural recording is bundled. "
                "Mathematical graph intervention is not an experimental "
                "cortical lesion. Dataset licenses and repeated-trial "
                "noise ceilings require verification by the operator."}
    if out:
        path=Path(out);path.mkdir(parents=True,exist_ok=True)
        (path/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-70")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23])
    a=p.parse_args()
    r=run(a.out,a.seeds)
    print(json.dumps({"status":r["evidence_status"],
                      "studies":len(r["parts"]),"claims":r["claims"]},indent=2))


if __name__=="__main__":
    main()
