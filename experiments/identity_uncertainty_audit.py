"""Small identity/uncertainty assay: no trained model or fake object permanence."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import identity_uncertainty_world as world


def run(out="runs/identity-uncertainty-audit.json",seeds=(11,23,41)):
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Need unique world seeds")
    data=[]
    for seed in seeds:
        first,second=world.paired_hidden_identity(int(seed))
        data.append({
            "seed":int(seed),
            "same_full_rgb_video":bool(np.array_equal(
                first["frames"],second["frames"])),
            "different_hidden_identity_after_occlusion":bool(np.any(
                first["positions"][8:]!=second["positions"][8:])),
            "identity_score_defined":world.identity_score(0,first),
            "uniform_branch_brier":.5*(world.branch_brier(.5,False)+
                                       world.branch_brier(.5,True)),
            "uniform_branch_log_loss":.5*(world.branch_log_loss(.5,False)+
                                          world.branch_log_loss(.5,True))
        })
    report={"status":"evaluator-level ambiguity audit; no model trained",
            "regime":"identical RGB but two different hidden identity assignments",
            "records":data,
            "warning":"No RGB model can identify a hidden swap invisible throughout the full video."}
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/identity-uncertainty-audit.json")
    a=p.parse_args()
    print(json.dumps(run(a.out),indent=2))


if __name__=="__main__":
    main()
