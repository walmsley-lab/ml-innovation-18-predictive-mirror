"""46: Select equally sized visual attention windows using prior-only evidence.

The selector consumes observed RGB context, never future flow/object truth.
Score whether the fixed sensing budget covers independently realized motion.
A handcrafted diagnostic, not yet a learned attention controller.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src import rgb_world_ladder as worlds

MODES=("surprise","salience","center","random")


def windows(size, width=6, stride=3):
    if width<1 or width>size or stride<1:
        raise ValueError("Invalid sensing window")
    return [(y,x) for y in range(0,size-width+1,stride)
            for x in range(0,size-width+1,stride)]


def choose(context, mode="surprise", count=3, width=6, seed=11):
    """Return window origins; future RGB and simulator truth not accepted."""
    context=np.asarray(context,float)
    if context.ndim!=4 or context.shape[-1]!=3 or len(context)<2:
        raise ValueError("Need two observed RGB images")
    h,w=context.shape[1:3]
    if h!=w or mode not in MODES:
        raise ValueError("Square RGB and known mode required")
    available=windows(h,width)
    if not 1<=count<=len(available):
        raise ValueError("Invalid attention budget")
    current=context[-1]
    score=(np.mean(np.abs(current-context[-2]),-1) if mode=="surprise"
           else np.max(current,axis=-1) if mode=="salience"
           else np.zeros((h,w)))
    if mode=="random":
        return [available[int(i)] for i in
                np.random.default_rng(seed).choice(len(available),count,replace=False)]
    if mode=="center":
        return sorted(available,key=lambda p:(
            (p[0]+width/2-h/2)**2+(p[1]+width/2-w/2)**2,p))[:count]
    return sorted(available,key=lambda p:(-float(np.sum(
        score[p[0]:p[0]+width,p[1]:p[1]+width])),p))[:count]


def mask(origins,size,width):
    out=np.zeros((size,size),bool)
    for y,x in origins:
        out[y:y+width,x:x+width]=True
    return out


def episode(seed,domain="occlusion_long",context=5,horizon=3,size=24):
    rgb=worlds.episode(seed,context+horizon,size,domain=domain)["frames"]
    reference=rgb[context-1]
    # Evaluation-only motion: never sent to choose.
    future=np.max(np.mean(np.abs(rgb[context:]-reference),axis=-1),axis=0)
    event=future>.06
    return rgb[:context],event


def run(out="runs/roadmap-46/metrics.json",seeds=range(91000,91032),
        domains=("linear","occlusion_long","accelerating","camera_pan"),
        count=3,width=6):
    rows=[]
    for domain in domains:
        for seed in seeds:
            past,event=episode(int(seed),domain=domain)
            for mode in MODES:
                sampled=mask(choose(past,mode,count,width,int(seed)),len(event),width)
                rows.append({"domain":domain,"seed":int(seed),"mode":mode,
                    "event_recall":float(np.logical_and(sampled,event).sum()/
                        max(int(event.sum()),1)),
                    "observed_pixel_fraction":float(sampled.mean()),
                    "event_pixels":int(event.sum())})
    report={"roadmap_id":46,"window_budget":count,"width":width,
            "rows":rows,"limits":"Uses past RGB surprise, not learned selective attention. "
             "Overlapping windows can vary unique covered pixels; both window "
             "and pixel budgets are reported. Future changes are scoring only."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runs/roadmap-46/metrics.json")
    a=p.parse_args()
    print(json.dumps({"cases":len(run(a.out)["rows"])}))
