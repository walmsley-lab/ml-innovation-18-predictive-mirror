"""48: bounded, observed-RGB working memory for colored moving objects.

Fixed marker colors identify objects *from their visible pixels*; there is no
RGB-learned identity. Only evaluator sees true hidden position and occlusion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src import rgb_robust_world

PALETTE=np.array([[.92,.19,.12],[.12,.70,.94],[.96,.75,.11],[.53,.27,.94]])


def detect(frame):
    """Fixed-color synthetic sensor baseline; no world-state input."""
    out={}
    h,w=frame.shape[:2]
    yy,xx=np.mgrid[:h,:w]
    for i,color in enumerate(PALETTE):
        near=np.max(np.abs(frame-color),axis=-1)<.07
        if near.sum()>=3:
            out[i]=np.array([float(xx[near].mean()),float(yy[near].mean())])
    return out


def update(memory,observations,t,capacity=2,policy="recency"):
    if policy not in ("recency","novelty","random") or capacity<1:
        raise ValueError("Bad memory policy or capacity")
    state={i:dict(value) for i,value in memory.items()}
    for identity,point in observations.items():
        previous=state.get(identity)
        velocity=(point-previous["position"])/(t-previous["seen"]) if (
            previous and t>previous["seen"]) else np.zeros(2)
        state[identity]={"position":point.copy(),"velocity":velocity,
                         "seen":t,"novelty":1 if previous is None else float(
                             np.linalg.norm(point-previous["position"]))}
    if len(state)>capacity:
        if policy=="recency":
            keep=sorted(state,key=lambda i:(-state[i]["seen"],i))[:capacity]
        elif policy=="novelty":
            keep=sorted(state,key=lambda i:(-state[i]["novelty"],-state[i]["seen"],i))[:capacity]
        else:
            keep=list(np.random.default_rng(t+111).permutation(sorted(state)))[:capacity]
        state={i:state[i] for i in keep}
    return state


def predict(memory,identity,t,velocity=True):
    if identity not in memory:
        return None
    state=memory[identity]
    gap=t-state["seen"]
    return state["position"]+(state["velocity"]*gap if velocity else 0.)


def run(out="runs/roadmap-48/metrics.json",seeds=range(91000,91020),
        capacity=2,steps=18,size=32):
    if capacity<1 or capacity>4:
        raise ValueError("Capacity must fit synthetic palette")
    rows=[]
    for seed in seeds:
        record=rgb_robust_world.episode(int(seed),steps=steps,size=size,
                                      objects=4,domain="train")
        for policy in ("recency","novelty","random"):
            memory={}
            errors=[];missing=0;events=0
            for t,frame in enumerate(record["frames"]):
                memory=update(memory,detect(frame),t,capacity,policy)
                for i in range(4):
                    # Explicitly evaluator-only: was in view but occluded?
                    if not record["occluded"][t,i]:
                        continue
                    events+=1
                    estimated=predict(memory,i,t)
                    if estimated is None:
                        missing+=1
                    else:
                        world_xy=(record["positions"][t,i]-
                                  record["camera_xy"][t]-size/2)
                        errors.append(float(np.linalg.norm(estimated-world_xy)))
            rows.append({"seed":int(seed),"policy":policy,"hidden_events":events,
                         "unrecalled_hidden":missing,"scorable_hidden":len(errors),
                         "hidden_mae_px":float(np.mean(errors)) if errors else None,
                         "capacity":capacity})
    report={"roadmap_id":48,"rows":rows,
            "limitations":"Synthetic color-coded identity, not learned RGB identity. "
            "Only evaluator reads occlusion and world positions. Zero hidden "
            "events produce a null error, never zero; capacity and memory "
            "retention are measured independently."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-48/metrics.json")
    print(json.dumps({"rows":len(run(p.parse_args().out)["rows"])}))
