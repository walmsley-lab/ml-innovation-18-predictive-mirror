"""Roadmap 33: bounded rehearsal versus recent-only continual learning.

Equal number of SGD updates for each observed example. Rehearsal samples
only earlier labeled observations from a fixed-size reservoir, never test
data or examples whose labels have not yet arrived.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def task(seed, name, count):
    if name not in ("A","B") or count < 1:
        raise ValueError("Unknown task or empty sample set")
    rng=np.random.default_rng(int(seed))
    if name=="A":
        value=rng.uniform(.1,1.,count)
        target=1.25*value+.15
    else:
        value=rng.uniform(-1.,-.1,count)
        target=-.8*value+.15
    features=np.stack((np.ones(count),value,np.maximum(value,0.)),axis=-1)
    return features,target+rng.normal(0,.025,count)


def update(weights,x,y,lr=.08):
    if lr <= 0:
        raise ValueError("Update rate must be positive")
    error=float(np.clip(np.dot(weights,x)-y,-3.,3.))
    return weights-lr*error*x


def reservoir_add(memory,element,seen,capacity,rng):
    """Unbiased online reservoir; element becomes eligible only next step."""
    if capacity < 1:
        raise ValueError("Replay memory must be positive")
    if len(memory) < capacity:
        memory.append(element)
    else:
        index=int(rng.integers(0,seen+1))
        if index < capacity:
            memory[index]=element
    return memory


def evaluate(weights,validation):
    return {name:float(np.square(x@weights-y).mean())
            for name,(x,y) in validation.items()}


def learn(seed=11,mode="replay",capacity=24,lr=.08,
          stages=(60,100,40)):
    if mode not in ("replay","recent_only") or capacity < 1 or min(stages)<1:
        raise ValueError("Invalid matched experiment")
    validation={name:task(91000+100*int(seed)+(0 if name=="A" else 50),
                          name,64) for name in ("A","B")}
    stream=[("A",stages[0]),("B",stages[1]),("A",stages[2])]
    rng=np.random.default_rng(int(seed)+40000)
    weights=np.zeros(3,float)
    memory=[]
    seen=0
    reused=0
    max_replay_index=-1
    curve=[]
    updates=0
    for stage,(name,count) in enumerate(stream):
        x,y=task(31000+100*int(seed)+stage,name,count)
        for observation,(feature,target) in enumerate(zip(x,y)):
            # Current supervision arrives after the prediction is issued.
            _committed_prediction=float(feature@weights)
            weights=update(weights,feature,target,lr)
            updates+=1
            if mode=="replay" and memory:
                location=int(rng.integers(len(memory)))
                old_index,old_x,old_y=memory[location]
                if old_index >= seen:
                    raise AssertionError("Attempt to train on unseen future")
                weights=update(weights,old_x,old_y,lr)
                reused+=1
                max_replay_index=max(max_replay_index,old_index)
            else:
                # Fair compute control: both arms spend TWO gradient steps
                # per current example, even when the replay memory is empty.
                weights=update(weights,feature,target,lr)
            updates+=1
            reservoir_add(memory,(seen,feature.copy(),float(target)),
                          seen,capacity,rng)
            seen+=1
        curve.append({"stage":stage,"task":name,"examples_seen":seen,
                      **evaluate(weights,validation)})
    return {"seed":int(seed),"mode":mode,"capacity":capacity,
            "total_examples":seen,"gradient_updates":updates,
            "replayed_updates":reused,"max_replay_index":max_replay_index,
            "stages":curve,"weights":weights}


def run(out="runs/roadmap-33",seeds=(11,23,41),capacity=24):
    if not seeds:
        raise ValueError("Need multiple independently initialized runs")
    folder=Path(out) if out is not None else None
    if folder is not None:
        folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        for mode in ("recent_only","replay"):
            r=learn(int(seed),mode,capacity)
            rows.append({k:v for k,v in r.items() if k!="weights"})
            if folder is not None:
                np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                    weights=r["weights"])
    summary={}
    for mode in ("recent_only","replay"):
        selected=[row for row in rows if row["mode"]==mode]
        summary[mode]={
            "A_after_A":float(np.mean([r["stages"][0]["A"] for r in selected])),
            "A_after_B":float(np.mean([r["stages"][1]["A"] for r in selected])),
            "B_after_B":float(np.mean([r["stages"][1]["B"] for r in selected])),
            "A_after_return":float(np.mean([r["stages"][2]["A"] for r in selected])),
            "B_after_return":float(np.mean([r["stages"][2]["B"] for r in selected])),
            "A_forgetting_delta":float(np.mean([
                r["stages"][1]["A"]-r["stages"][0]["A"] for r in selected]))}
    result={"roadmap_id":33,"seeds":list(map(int,seeds)),
            "memory_capacity":capacity,"gradient_steps_per_example":2,
            "test":summary,"per_seed":rows,
            "limits":"Task-specific observed scalar feature distributions and conflicting "
                     "regressions, not vision or an RGB NCA replay buffer. "
                     "Held-out evaluation is never used for gradient updates. "
                     "Rehearsal has memory and read costs; compute steps are matched. "
                     "Retain both plasticity/forgetting tradeoffs, not one leaderboard."}
    if folder is not None:
        (folder/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-33")
    p.add_argument("--capacity",type=int,default=24)
    a=p.parse_args()
    print(json.dumps(run(a.out,capacity=a.capacity)["test"],indent=2))


if __name__=="__main__":
    main()
