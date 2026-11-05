"""54: active two-dimensional gaze under a strict visual sensing budget.

Each policy receives exactly ONE 3x3 measurement per step. Actions are
chosen from the posterior belief based on past observations, not hidden
target locations. Fixed-policy parameters may use a separate training set.
"""
import argparse
import json
from pathlib import Path
import numpy as np

SIZE=8
WINDOW=3
ORIGINS=[(y,x) for y in range(SIZE-WINDOW+1)
         for x in range(SIZE-WINDOW+1)]
POLICIES=("active","fixed","random")


def world(seed,steps=24):
    rng=np.random.default_rng(int(seed))
    current=rng.integers(0,SIZE,size=2)
    path=[]
    for _ in range(steps):
        path.append(current.copy())
        current=np.clip(current+rng.integers(-1,2,size=2),0,SIZE-1)
    return np.asarray(path)


def propagate(belief):
    if belief.shape!=(SIZE,SIZE):
        raise ValueError("Belief must be 8x8")
    out=np.zeros_like(belief,dtype=float)
    for y in range(SIZE):
        for x in range(SIZE):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    oy=int(np.clip(y+dy,0,SIZE-1))
                    ox=int(np.clip(x+dx,0,SIZE-1))
                    out[oy,ox]+=belief[y,x]/9.
    return out/out.sum()


def choose(belief,prior=None):
    """Maximize predicted probability of seeing the target; no oracle."""
    choices=[]
    for y,x in ORIGINS:
        mass=belief[y:y+WINDOW,x:x+WINDOW].sum()
        travel=0 if prior is None else np.linalg.norm(np.array([y,x])-prior)
        choices.append((-float(mass),float(travel),y,x))
    best=min(choices)
    return best[2],best[3]


def observe(belief,gaze,target):
    y,x=gaze
    visible=y<=target[0]<y+WINDOW and x<=target[1]<x+WINDOW
    result=belief.copy()
    if visible:
        result[:]=0
        result[target[0],target[1]]=1.
    else:
        result[y:y+WINDOW,x:x+WINDOW]=0
        total=result.sum()
        if total<1e-12:
            raise ValueError("Impossible observation for this belief")
        result/=total
    return result,bool(visible)


def fixed_training(seeds=range(31000,31020)):
    # Optimistic fixed baseline: optimize occupancy with TRAIN labels only.
    occupancy=np.zeros((SIZE,SIZE),int)
    for seed in seeds:
        for y,x in world(int(seed)):
            occupancy[y,x]+=1
    return choose(occupancy/occupancy.sum())


def episode(seed,policy,fixed=(2,2),steps=24):
    if policy not in POLICIES or fixed not in ORIGINS:
        raise ValueError("Invalid gaze policy")
    truth=world(seed,steps)
    belief=np.full((SIZE,SIZE),1/(SIZE*SIZE),float)
    rng=np.random.default_rng(int(seed)+222)
    previous=None
    errors=[];hits=[];travel=[]
    for t,point in enumerate(truth):
        if t:
            belief=propagate(belief)
        gaze=(choose(belief,previous) if policy=="active" else
              fixed if policy=="fixed" else ORIGINS[int(rng.integers(len(ORIGINS)))])
        belief,hit=observe(belief,gaze,point)
        yy,xx=np.mgrid[:SIZE,:SIZE]
        estimate=np.array([(belief*yy).sum(),(belief*xx).sum()])
        errors.append(float(np.linalg.norm(estimate-point)))
        hits.append(int(hit))
        travel.append(0. if previous is None else float(
            np.linalg.norm(np.array(gaze)-previous)))
        previous=gaze
    return {"position_mae_cells":float(np.mean(errors)),
            "sighting_rate":float(np.mean(hits)),
            "gaze_distance_per_step":float(np.mean(travel)),
            "observations":steps}


def run(out="runs/roadmap-54/metrics.json",seeds=range(91000,91032)):
    fixed=fixed_training()
    rows=[{"seed":int(seed),"policy":mode,
           **episode(int(seed),mode,fixed)}
          for seed in seeds for mode in POLICIES]
    report={"roadmap_id":54,"fixed_gaze":list(fixed),"rows":rows,
            "limits":"One equally sized visual field per step, exact-in-window "
             "target detection and known motion transition. Active Bayesian "
             "policy not learned neural attention; training-optimized fixed "
             "floor is deliberately strong. Gaze movement is extra cost."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-54/metrics.json")
    print(json.dumps({"rows":len(run(p.parse_args().out)["rows"])}))
