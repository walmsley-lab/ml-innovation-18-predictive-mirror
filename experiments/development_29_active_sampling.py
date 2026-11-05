"""Roadmap 29: information-directed gaze with equal observation counts.

A tiny 1D partially observed world: gaze sees one window per time step.
The policy receives only its causal belief. Hidden target positions are used
by the sensor/evaluator, never to choose the next window.
"""
import argparse
import json
from pathlib import Path
import numpy as np

WIDTH=21
WINDOW=5
WINDOWS=tuple(range(0,WIDTH-WINDOW+1))


def transition(belief):
    """Reflecting, known three-step state transition reference."""
    n=len(belief)
    nxt=np.zeros(n,float)
    for i,prob in enumerate(belief):
        for delta,weight in ((-1,.35),(0,.30),(1,.35)):
            target=min(n-1,max(0,i+delta))
            nxt[target]+=prob*weight
    return nxt


def entropy(prob):
    p=np.asarray(prob)
    p=p[p>0]
    return float(-(p*np.log(p)).sum())


def expected_entropy(belief, start, window=WINDOW):
    """A positive sighting reveals the exact in-window coordinate."""
    outside=belief.copy()
    outside[start:start+window]=0.
    mass=float(outside.sum())
    return mass*entropy(outside/mass) if mass>0 else 0.


def choose(belief):
    """No hidden trajectory enters this decision."""
    return min(WINDOWS, key=lambda start: (expected_entropy(belief,start),start))


def update(belief, start, observation):
    posterior=belief.copy()
    if observation is None:
        posterior[start:start+WINDOW]=0.
    else:
        if not start<=observation<start+WINDOW:
            raise ValueError("Sighting lies outside gaze")
        posterior[:]=0.
        posterior[observation]=1.
    total=posterior.sum()
    if total<=0:
        raise ValueError("Impossible measurement under predicted belief")
    return posterior/total


def world(seed, steps=24):
    rng=np.random.default_rng(seed)
    x=int(rng.integers(0,WIDTH))
    path=[x]
    for _ in range(steps-1):
        x=int(np.clip(x+rng.choice((-1,0,1),p=(.35,.3,.35)),0,WIDTH-1))
        path.append(x)
    return path


def episode(seed,policy,fixed=8,steps=24):
    if policy not in ("active","fixed","random"):
        raise ValueError("Unknown gaze policy")
    truth=world(seed,steps)
    belief=np.full(WIDTH,1./WIDTH)
    rng=np.random.default_rng(seed+999991)
    rows=[]; last=None
    for t,x in enumerate(truth):
        if t:
            belief=transition(belief)
        start=(choose(belief) if policy=="active" else
               fixed if policy=="fixed" else int(rng.choice(WINDOWS)))
        if start not in WINDOWS:
            raise ValueError("Invalid fixed gaze")
        observed=x if start<=x<start+WINDOW else None
        belief=update(belief,start,observed)
        rows.append({"entropy":entropy(belief),
                     "position_mae":float(abs(np.dot(np.arange(WIDTH),belief)-x)),
                     "sighting":int(observed is not None),
                     "gaze_movement":0 if last is None else abs(start-last)})
        last=start
    return {key:float(np.mean([row[key] for row in rows]))
            for key in rows[0]}


def select_fixed(train_seeds):
    """Fit a fixed policy ONLY on training episodes, not test states."""
    averages={s:float(np.mean([episode(seed,"fixed",s)["position_mae"]
                                for seed in train_seeds])) for s in WINDOWS}
    return min(averages,key=lambda k:(averages[k],k))


def run(out="runs/development-29/metrics.json",
        train_seeds=range(31000,31032),test_seeds=range(91000,91032)):
    fixed=select_fixed(train_seeds)
    scores={}
    for name in ("active","fixed","random"):
        rows=[episode(seed,name,fixed) for seed in test_seeds]
        scores[name]={key:float(np.mean([row[key] for row in rows]))
                      for key in rows[0]}
    report={"roadmap_id":29,"optimized_fixed_window":int(fixed),
            "train_episodes":len(train_seeds),"test_episodes":len(test_seeds),
            "steps_per_episode":24,"observations_per_step":1,"test":scores,
            "limits":"Bayesian motion prior and exact in-window location sensing; "
                     "information-gain heuristic, not learned neural attention. "
                     "Window movement costs reported separately; equal visual "
                     "samples does not imply equal motor cost."}
    if out is not None:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/development-29/metrics.json")
    a=p.parse_args()
    print(json.dumps(run(a.out),indent=2))


if __name__=="__main__":
    main()
