"""49: operational object lifecycle from RGB without hidden death labels.

New/visible/temporarily missing/retired/reappeared are inferred only from
observed colored blobs. "Retired" means absence beyond a threshold, not
proof that the physical object ceased to exist.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.roadmap_48_working_memory import detect, PALETTE

STATES=("unseen","visible","uncertain","retired","reappeared")


def episode(seed,steps=16,size=24):
    if steps<14 or size<20:
        raise ValueError("Need birth, death and occlusion events")
    rng=np.random.default_rng(seed)
    frames=np.full((steps,size,size,3),.035,np.float32)
    alive=np.zeros((steps,2),bool)
    occluded=np.zeros_like(alive)
    for t in range(steps):
        alive[t,0]=2<=t<=11
        alive[t,1]=6<=t<=steps-2
        occluded[t,0]=t in (5,6,7)
        occluded[t,1]=t in (9,10)
        for j in range(2):
            if not alive[t,j] or occluded[t,j]:
                continue
            x=5+int((2*t+j*3)%12)
            y=7+j*7+int(rng.integers(-1,2))
            frames[t,y-1:y+2,x-1:x+2]=PALETTE[j]
    return frames,alive,occluded


def update(state,observations,t,retire_after=3):
    if retire_after<1:
        raise ValueError("retirement requires positive absence window")
    result={k:dict(v) for k,v in state.items()}
    for i in set(result)|set(observations):
        detected=i in observations
        prior=result.get(i,{"last_seen":-1,"state":"unseen"})
        if detected:
            lifecycle=("reappeared" if prior["state"] in ("uncertain","retired")
                       else "visible")
            result[i]={"state":lifecycle,"last_seen":t}
        elif prior["last_seen"]>=0:
            age=t-prior["last_seen"]
            result[i]={"state":"retired" if age>retire_after else "uncertain",
                       "last_seen":prior["last_seen"]}
    return result


def trajectory(frames,retire_after=3):
    state={}
    history=[]
    for t,image in enumerate(frames):
        state=update(state,detect(image),t,retire_after)
        history.append({i:entry["state"] for i,entry in state.items()})
    return history


def run(out="runs/roadmap-49/metrics.json",seeds=range(91000,91030)):
    rows=[]
    for seed in seeds:
        frames,alive,occluded=episode(int(seed))
        for threshold in (1,3,5):
            states=trajectory(frames,threshold)
            false_retired=0;total_hidden=0;undetected=0;revived=0
            for t in range(len(frames)):
                for i in range(2):
                    label=states[t].get(i,"unseen")
                    if alive[t,i] and occluded[t,i]:
                        total_hidden+=1
                        false_retired+=int(label=="retired")
                    if alive[t,i] and not occluded[t,i]:
                        undetected+=int(label in ("unseen","retired"))
                    revived+=int(label=="reappeared")
            rows.append({"seed":int(seed),"threshold":threshold,
                         "false_retire_during_occlusion":false_retired,
                         "occluded_alive_events":total_hidden,
                         "missed_visible_events":undetected,
                         "reappearance_transitions":revived})
    report={"roadmap_id":49,"rows":rows,
            "limits":"Visible RGB color markers identify objects; test birth/death "
             "truth and occlusion masks are scoring only. Impossible to infer "
             "physical death uniquely from absent pixels; retirement is a "
             "conservative operational state."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-49/metrics.json")
    print(json.dumps({"rows":len(run(p.parse_args().out)["rows"])}))
