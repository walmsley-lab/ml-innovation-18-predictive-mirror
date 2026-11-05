"""Frozen privileged-state Gaussians: static vs velocity vs future oracle."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import world3d,gaussian_splats


def run(out="runs/gaussian-oracle-baselines/metrics.json",seeds=(11,23,41),
        context=5,horizon=5,size=24,domain="ood_camera"):
    if not seeds or context<2 or horizon<1:
        raise ValueError("Invalid oracle baseline")
    rows=[]
    for seed in seeds:
        world=world3d.episode(seed,context+horizon,size,domain)
        records=gaussian_splats.evaluate(world,context,horizon)
        rows.append({"seed":int(seed),"modes":{
            name:{k:v for k,v in values.items() if k!="frames"}
            for name,values in records.items()}})
    report={"protocol":"privileged-gaussian-scene-controls-v1",
            "seeds":list(map(int,seeds)),"context":context,"horizon":horizon,
            "size":size,"domain":domain,"rows":rows,
            "privileged_inputs":"Simulated metric centers, radii, colors, "
             "observed positional velocity. Future centers only in oracle arm.",
            "model_input_contract":"Future work must infer persistent state "
             "from observed RGB and may use predeclared camera actions; "
             "no future scene state, RGB or depth supplied at inference.",
            "limits":"Gaussian and analytic ray-sphere renderers differ, so "
             "the oracle's RGB error is a representation gap, not zero. "
             "Static/velocity use true observed geometry and are privileged "
             "controls. No neural training, object discovery or proof of "
             "learned 3D perception."}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/gaussian-oracle-baselines/metrics.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--context",type=int,default=5)
    p.add_argument("--horizon",type=int,default=5)
    p.add_argument("--size",type=int,default=24)
    p.add_argument("--domain",choices=world3d.DOMAINS,default="ood_camera")
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.context,a.horizon,a.size,a.domain)
                     ["rows"],indent=2))


if __name__=="__main__":
    main()
