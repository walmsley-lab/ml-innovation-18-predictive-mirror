"""Verify layered-world truth and emit compact render/flow samples."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import layered_advection_world as world


def run(out="runs/layered-advection",seeds=(11,23,41),modes=world.MODES,
        size=32,steps=12):
    if not seeds:
        raise ValueError("Need independent seed episodes")
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for mode in modes:
        for seed in seeds:
            rec=world.episode(seed,steps,size,mode)
            checks=world.validate(rec)
            name=f"{mode}-seed{seed}.npz"
            np.savez_compressed(root/name,**{
                key:value for key,value in rec.items() if isinstance(value,np.ndarray)})
            rows.append({"seed":int(seed),"mode":mode,"example":name,
                         "validation":checks})
    summary={"protocol":"layered-advection-truth-v1","rows":rows,
             "limits":"Optical flow, instance IDs, depth ordering, about-to-occlude "
              "and source/target visibility are evaluator-only. "
              "Rasterized visibility uses nearest-pixel correspondence "
              "for fractional motion. This is not yet a realistic natural-video "
              "entropy distribution or a learned predictor."}
    (root/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/layered-advection")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    a=p.parse_args()
    print(json.dumps({"episodes":len(run(a.out,a.seeds)["rows"])}))


if __name__=="__main__":
    main()
