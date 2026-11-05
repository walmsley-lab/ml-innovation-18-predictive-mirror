"""50: bounded multi-hypothesis correspondence from visual detections alone.

No simulator identity or hidden coordinate enters the tracker; truth may
score identity errors only after a history-conditioned forecast is issued.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from experiments import development_24_correspondence as reference


def proposals(positions,velocity,detections,max_gap=8):
    predicted=positions+velocity
    n=len(predicted)
    if n!=2 or len(detections)>2:
        raise ValueError("Two visual tracks supported")
    # Each observed blob is either one track's measurement or unassigned.
    for assigned in itertools.product(range(-1,len(detections)),repeat=n):
        used=[a for a in assigned if a>=0]
        if len(used)!=len(set(used)):
            continue
        estimate=predicted.copy();rates=velocity.copy()
        cost=0.
        for i,index in enumerate(assigned):
            if index<0:
                cost+=6.
            else:
                diff=detections[index]-predicted[i]
                cost+=float(diff@diff)
                estimate[i]=detections[index]
                rates[i]=.7*velocity[i]+.3*(estimate[i]-positions[i])
        cost+=6.*(len(detections)-len(used))
        yield estimate,rates,cost


def track(frames,beam=3):
    if beam<1:
        raise ValueError("Need positive hypothesis count")
    first=reference.detections(frames[0])
    if len(first)!=2:
        raise ValueError("Require two distinct observed sources initially")
    position=first[np.argsort(first[:,0])]
    hypotheses=[(0.,position,np.zeros_like(position),[position.copy()])]
    for frame in frames[1:]:
        found=reference.detections(frame)
        next_hypotheses=[]
        for old_cost,positions,velocity,path in hypotheses:
            for p,v,extra in proposals(positions,velocity,found):
                next_hypotheses.append((old_cost+extra,p,v,path+[p.copy()]))
        hypotheses=sorted(next_hypotheses,key=lambda a:a[0])[:beam]
    return np.stack(hypotheses[0][3])


def score(predicted,truth,visible):
    if predicted.shape!=truth.shape:
        raise ValueError("Prediction/truth mismatch")
    error=np.linalg.norm(predicted-truth,axis=-1)
    return {"visible_mae_px":float(error[visible].mean()),
            "hidden_mae_px":float(error[~visible].mean()) if (~visible).any() else None,
            "post_crossing_mae_px":float(error[6:].mean())}


def run(out="runs/roadmap-50/metrics.json",seeds=range(91000,91032)):
    rows=[]
    for seed in seeds:
        frames,truth,visible=reference.world(int(seed))
        for arm in ("beam1","beam3","nearest","coast"):
            prediction=(track(frames,1 if arm=="beam1" else 3)
                        if arm.startswith("beam") else reference.track(frames,arm))
            rows.append({"seed":int(seed),"arm":arm,
                         "scores":score(prediction,truth,visible)})
    result={"roadmap_id":50,"rows":rows,
            "limitations":"Same-color pixel detections are supplied by a threshold "
            "detector; identity is initialized by left-to-right order. "
            "Beam search has additional inference work, may still swap IDs. "
            "No pose or pixel segmentation oracle reaches the tracker."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-50/metrics.json")
    print(json.dumps({"cases":len(run(p.parse_args().out)["rows"])}))
