"""Frozen NCA one-second occlusion and two-ball interaction stress test."""
import argparse
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark


def reappearance_count(records, context, horizon):
    """Evaluation-only: count visible returns after an unobserved context frame."""
    count=0
    for record in records:
        seen=np.asarray(record["visible"])[:,0]
        hidden=not bool(seen[context-1])
        if hidden and any(bool(x) for x in seen[context:context+horizon]):
            count+=1
    return count


def evaluate(checkpoint, output="runs/rgb-occlusion/metrics.json",
             seed=82000, count=24, context=8, horizon=10, ticks=2,
             size=32):
    if context<2 or horizon<1 or count<1:
        raise ValueError("Invalid test configuration")
    with np.load(checkpoint,allow_pickle=False) as archive:
        params={k:jnp.asarray(archive[k]) for k in archive.files}
    scenarios={"clear":{"balls":1},
               "occluded":{"balls":1,"occlusion":True},
               "two_balls":{"balls":2}}
    leads=tuple(h for h in benchmark.LEADS if h<=horizon)
    rows={}
    for index,(name,world) in enumerate(scenarios.items()):
        records=benchmark.scenes(seed+index*1000,count,context,horizon,
                                 size,**world)
        x=benchmark.frames(records)
        predicted=np.asarray(nca.rollout(
            params,jnp.asarray(x[:,:context]),horizon,ticks))
        controls=benchmark.controls(x[:,:context],horizon)
        scores={"nca":benchmark.score(predicted,x,context,leads)}
        scores.update({k:benchmark.score(v,x,context,leads)
                       for k,v in controls.items()})
        rows[name]={"scores":scores,"episodes":count}
        if name=="occluded":
            rows[name]["reappearance_episodes"]=reappearance_count(
                records,context,horizon)
    report={"experiment":"frozen_rgb_occlusion_interactions",
            "checkpoint":str(checkpoint),"context":context,"horizon":horizon,
            "seed_start":seed,"scenarios":rows,
            "limitations":"Reappearance counts use simulator visibility for "
                          "post-hoc event classification only. The RGB model "
                          "sees neither world state nor future observations. "
                          "Random scenes may yield zero reappearance events, "
                          "in which case that ability is not evaluated."}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",default="runs/rgb-occlusion/metrics.json")
    p.add_argument("--seed",type=int,default=82000)
    p.add_argument("--episodes",type=int,default=24)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--size",type=int,default=32)
    a=p.parse_args()
    result=evaluate(a.checkpoint,a.out,a.seed,a.episodes,a.context,
                    a.horizon,a.ticks,a.size)
    print(json.dumps(result["scenarios"],indent=2))


if __name__=="__main__":
    main()
