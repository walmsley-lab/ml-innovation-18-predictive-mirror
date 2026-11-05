"""52: surprise-triggered recovery, with prediction-before-observation.

A scalar motion state predicts the next observation; corrections use *only*
labels that have arrived. Sudden velocity changes and sensor glitches are
separate events. This is a state-estimator reference, not a learned NCA.
"""
import argparse
import json
from pathlib import Path
import numpy as np

MODES=("fixed","always_correct","bounded_innovation","delayed_feedback")


def world(seed=11,steps=32,kind="regime"):
    if steps<25 or kind not in ("regime","sensor_glitch","both"):
        raise ValueError("Need events and sufficient recovery interval")
    rng=np.random.default_rng(int(seed))
    velocity=1.3
    position=0.
    observations=[]
    for t in range(steps):
        if t==12 and kind in ("regime","both"):
            velocity=-.9
        position+=velocity
        value=position+rng.normal(0,.12)
        if t==20 and kind in ("sensor_glitch","both"):
            value+=15.
        observations.append(value)
    return np.asarray(observations)


def infer(observations,mode="bounded_innovation",threshold=2.5):
    if mode not in MODES or len(observations)<3 or threshold<=0:
        raise ValueError("Bad prequential estimator")
    pos=float(observations[1])
    velocity=float(observations[1]-observations[0])
    committed=[];innovations=[]
    old=None
    for t in range(2,len(observations)):
        prior=pos+velocity
        committed.append(prior)  # issued before reading observation[t]
        target=float(observations[t])
        innovation=target-prior
        innovations.append(innovation)
        if mode=="fixed":
            pos=prior
        elif mode=="delayed_feedback":
            used=0. if old is None else old
            pos=prior+.35*np.clip(used,-threshold,threshold)
            velocity+=.2*np.clip(used,-threshold,threshold)
        elif mode=="bounded_innovation":
            used=np.clip(innovation,-threshold,threshold)
            pos=prior+.35*used
            velocity+=.2*used
        else:
            pos=target
            velocity=.55*velocity+.45*(target-(prior-velocity))
        old=innovation
    return np.asarray(committed),np.asarray(innovations)


def score(predicted,truth):
    error=np.square(predicted-truth[2:])
    return {"pre_event_mse":float(error[:10].mean()),
            "event_first3_mse":float(error[10:13].mean()),
            "recovery_next5_mse":float(error[13:18].mean()),
            "glitch_window_mse":float(error[18:21].mean()),
            "overall_mse":float(error.mean())}


def run(out="runs/roadmap-52/metrics.json",seeds=range(91000,91032)):
    rows=[]
    for seed in seeds:
        for kind in ("regime","sensor_glitch","both"):
            observations=world(int(seed),kind=kind)
            for mode in MODES:
                forecast,_=infer(observations,mode)
                rows.append({"seed":int(seed),"domain":kind,"mode":mode,
                             "scores":score(forecast,observations)})
    result={"roadmap_id":52,"rows":rows,
            "limits":"Observed scalar motion rather than learned RGB representation. "
             "Predictions scored before updates, including at the event. "
             "Innovation clipping trades rapid genuine-shift adaptation "
             "against noise/outlier robustness. No hidden event labels "
             "are input to the estimator."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-52/metrics.json")
    print(json.dumps({"rows":len(run(p.parse_args().out)["rows"])}))
