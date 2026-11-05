"""Evaluate causal classical forecasts on independent synthetic episodes."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import rgb_motion_world as world
from src import classical_transport as control
from src.rgb_benchmark import score

MODES=("persistence","pixel_velocity","flow_gather","coast","reid")


def run(out="runs/classical-transport/metrics.json",seeds=(11,23,41),
        context=5,horizon=5,size=32):
    if not seeds or context<2 or horizon<1:
        raise ValueError("Invalid evaluation")
    episodes=[world.episode(int(s),steps=context+horizon,size=size,balls=1,
                             occlusion=True) for s in seeds]
    frames=np.stack([e["frames"] for e in episodes])
    rows={}
    per_scene={}
    for name in MODES:
        forecasts=np.stack([control.forecast(r["frames"][:context],
                                         horizon,name) for r in episodes])
        rows[name]=score(forecasts,frames,context,tuple(range(1,horizon+1)))
        per_scene[name]=np.square(forecasts-frames[:,context:]).mean(
            axis=(2,3,4)).tolist()
    # Joint error incidence on identical episodes; no pseudo-seed variance.
    coast=np.asarray(per_scene["coast"])
    overlaps={}
    for name in MODES:
        errors=np.asarray(per_scene[name])
        overlaps[name]={"both_bad":int(np.sum((errors>.02)&(coast>.02))),
                        "control_only_bad":int(np.sum((errors>.02)&(coast<=.02))),
                        "coast_only_bad":int(np.sum((errors<=.02)&(coast>.02))),
                        "both_good":int(np.sum((errors<=.02)&(coast<=.02))),
                        "error_correlation":float(np.corrcoef(
                            errors.flatten(),coast.flatten())[0,1])
                            if np.std(errors)>1e-12 and np.std(coast)>1e-12 else None}
    report={"protocol":"classical-causal-transport-v1","seeds":list(map(int,seeds)),
            "context":context,"horizon":horizon,"modes":rows,
            "per_episode_frame_mse":per_scene,"overlap_vs_coast":overlaps,
            "failure_threshold_mse":.02,
            "limits":"Fixed known synthetic palette is a disclosed detector prior. "
                     "All controls use past frames only. Coast/reid are basic "
                     "centroid filters, not pretrained object tracking. "
                     "Relative skill needs an absolute denominator floor. "
                     "An RGB transport warp does not synthesize disoccluded content."}
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/classical-transport/metrics.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds)["overlap_vs_coast"],indent=2))


if __name__=="__main__":
    main()
