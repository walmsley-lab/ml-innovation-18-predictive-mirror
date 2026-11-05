"""51: irregular camera timing in object absence and memory expiry.

A model sees only currently delivered RGB and its monotonic timestamp. The
evaluator knows whether a missing object is still alive but occluded.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import roadmap_49_lifecycle as lifecycle
from experiments.roadmap_48_working_memory import detect


def timestamps(seed,steps,mode):
    rng=np.random.default_rng(seed)
    if mode=="regular":
        dt=np.full(steps-1,.15)
    elif mode=="jitter":
        dt=rng.uniform(.07,.27,size=steps-1)
    elif mode=="burst":
        dt=rng.uniform(.07,.18,size=steps-1)
        dt[3::5]+=.43
    else:
        raise ValueError("Unknown timing mode")
    return np.r_[0.,np.cumsum(dt)]


def infer(frames,stamps,seconds=.45,frames_limit=3,mode="time"):
    if len(frames)!=len(stamps) or len(frames)<2 or (
            np.any(np.diff(stamps)<=0)) or mode not in ("time","frames"):
        raise ValueError("Requires aligned monotonic RGB/timestamps")
    memory={}
    states=[]
    for t,image in enumerate(frames):
        seen=detect(image)
        for identity in set(memory)|set(seen):
            before=memory.get(identity)
            if identity in seen:
                state="reappeared" if before and before["state"]=="retired" else "visible"
                memory[identity]={"state":state,"last_time":stamps[t],"last_frame":t}
            elif before:
                expired=((stamps[t]-before["last_time"]>seconds)
                         if mode=="time" else t-before["last_frame"]>frames_limit)
                memory[identity]={**before,"state":"retired" if expired else "uncertain"}
        states.append({k:v["state"] for k,v in memory.items()})
    return states


def score(states,alive,occluded):
    false_retired=0;true_absent_active=0;missed_reappearance=0
    for t,entry in enumerate(states):
        for i in range(alive.shape[1]):
            status=entry.get(i,"unseen")
            false_retired+=int(alive[t,i] and occluded[t,i] and status=="retired")
            true_absent_active+=int(not alive[t,i] and status in
                                    ("visible","uncertain","reappeared"))
            missed_reappearance+=int(alive[t,i] and not occluded[t,i] and
                                      status in ("unseen","retired"))
    return {"false_death_during_occlusion":false_retired,
            "unsupported_presence_after_death":true_absent_active,
            "missed_visible_object":missed_reappearance}


def run(out="runs/roadmap-51/metrics.json",seeds=range(91000,91024)):
    rows=[]
    for seed in seeds:
        frames,alive,occluded=lifecycle.episode(int(seed))
        for timing in ("regular","jitter","burst"):
            stamps=timestamps(int(seed),len(frames),timing)
            for mode in ("time","frames"):
                inferred=infer(frames,stamps,mode=mode)
                rows.append({"seed":int(seed),"timing":timing,"mode":mode,
                             "metrics":score(inferred,alive,occluded),
                             "duration_s":float(stamps[-1])})
    result={"roadmap_id":51,"rows":rows,
            "limits":"Synthetic timestamps and color-known object IDs. "
             "Age-based uncertainty is not proof of disappearance or "
             "capture timing under live webcam hardware. All statuses "
             "are computed from acquired past RGB and timestamps."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-51/metrics.json")
    print(json.dumps({"rows":len(run(p.parse_args().out)["rows"])}))
