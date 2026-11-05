"""47: Does spatial attention buy accuracy per updated pixel?

A classical image-motion extrapolation is applied only inside selected regions.
All policies receive the same available RGB history. Future truth is scoring
only; 'full velocity' uses more updates and is not resource-matched.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import roadmap_46_attention as attention
from src import rgb_benchmark as benchmark, rgb_world_ladder as world

MODES=("surprise","salience","center","random","full_velocity","persistence")


def one(frames,context=5,horizon=3,width=6,count=3,seed=11):
    x=np.asarray(frames,float)
    if x.ndim!=4 or len(x)<context+horizon or context<2:
        raise ValueError("Need observed RGB history and held-out future")
    controls=benchmark.controls(x[None,:context],horizon)
    velocity=controls["constant_velocity"][0]
    hold=controls["persistence"][0]
    result={}
    for mode in MODES:
        if mode in ("persistence","full_velocity"):
            selected=np.full(x.shape[1:3],mode=="full_velocity")
        else:
            selected=attention.mask(attention.choose(
                x[:context],mode,count,width,seed),x.shape[1],width)
        forecast=np.where(selected[None,:,:,None],velocity,hold)
        truth=x[context:context+horizon]
        squared=np.square(forecast-truth)
        changed=np.mean(np.abs(truth-x[context-1]),axis=-1)>.03
        result[mode]={
            "mse":float(squared.mean()),
            "motion_mse":float(np.mean(squared.mean(-1)[changed])) if changed.any() else None,
            "unique_updated_pixels":int(selected.sum()),
            "pixel_fraction":float(selected.mean())}
    return result


def run(out="runs/roadmap-47/metrics.json",seeds=range(91000,91016),
        domains=("linear","accelerating","occlusion_long","camera_pan")):
    rows=[]
    for domain in domains:
        for seed in seeds:
            frames=world.episode(int(seed),8,24,domain=domain)["frames"]
            rows.append({"seed":int(seed),"domain":domain,
                         "arms":one(frames,seed=int(seed))})
    result={"roadmap_id":47,"rows":rows,
            "limits":"Classical masked motion forecast, not learned recurrent "
            "attention. Full-velocity arm spends more pixel updates; fixed "
            "windows may overlap and unique pixels are recorded. "
            "Do not treat an area proxy as measured GPU energy."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-47/metrics.json")
    print(json.dumps({"episodes":len(run(p.parse_args().out)["rows"])}))
