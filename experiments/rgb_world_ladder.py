"""Freeze and inspect richer perceptual worlds before model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import rgb_world_ladder as world, rgb_benchmark, rgb_event_metrics


def run(out="runs/rgb-world-ladder/metrics.json",domains=world.DOMAINS,
        episodes=12,context=8,horizon=10,size=32,balls=2):
    if episodes<1 or context<2 or horizon<1:
        raise ValueError("Invalid frozen world comparison")
    result={}
    leads=tuple(sorted({horizon,*(i for i in (1,3,5,10) if i<=horizon)}))
    for domain in domains:
        records=world.split("test",episodes,context,horizon,size,balls,domain)
        frames=np.stack([r["frames"] for r in records])
        controls=rgb_benchmark.controls(frames[:,:context],horizon)
        result[domain]={
            "visibility_reasons":{
                category:int(np.count_nonzero(np.asarray([
                    r["visibility_reason"][context:] for r in records])==category))
                for category in ("visible","barrier","out_of_view")},
            "collisions":int(sum(r["collision_counts"].sum() for r in records)),
            "baselines":{k:rgb_event_metrics.score(v,records,context,leads)
                         for k,v in controls.items()}}
    report={"protocol":"rgb-perception-ladder-v1","seed_split":"test:91000+",
            "domains":list(domains),"episodes_per_domain":episodes,
            "context":context,"horizon":horizon,"size":size,"balls":balls,
            "results":result,
            "limits":"Simple discs, collisions are approximate. Fixed palette and "
                     "background contrast preserve the v1 event-metric contract; "
                     "the appearance domain changes size/background, not hue. "
                     "These are observation-only baselines: models must never "
                     "read positions, velocities, visibility or future frames."}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-world-ladder/metrics.json")
    p.add_argument("--episodes",type=int,default=12)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--balls",type=int,default=2)
    p.add_argument("--domains",nargs="+",choices=world.DOMAINS,default=list(world.DOMAINS))
    a=p.parse_args()
    print(json.dumps(run(a.out,a.domains,a.episodes,a.context,
                         a.horizon,a.size,a.balls)["domains"],indent=2))


if __name__=="__main__":
    main()
