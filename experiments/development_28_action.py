"""Roadmap 28: train a small action-conditioned observation-to-motion predictor.

A model sees observed RGB centroids and only camera commands known for the
forecast interval. The next frame is a training/scoring label, never an
inference feature. Train/action-blind arms have equal coefficient dimensions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src import rgb_world_ladder as worlds


def center(rgb):
    mask=(rgb[...,0]>.55)&(rgb[...,1]<.5)&(rgb[...,2]<.5)
    points=np.argwhere(mask)
    if not len(points):
        raise ValueError("Visible red object required")
    return points.mean(axis=0)[::-1]


def episode(seed, steps=9, size=32):
    return worlds.episode(seed,steps,size,balls=1,domain="camera_pan")


def feature(record, t, conditioned=True):
    """t>=1: predict observation at t+1, before image[t+1] arrives."""
    if not 1 <= t < len(record["frames"])-1:
        raise ValueError("Invalid forecast boundary")
    now=center(record["frames"][t])
    prev=center(record["frames"][t-1])
    action=np.asarray(record["camera_actions"],float)
    prev_action=action[t-1] if conditioned else np.zeros(2)
    next_action=action[t] if conditioned else np.zeros(2)
    return np.concatenate(([1.], now-prev, prev_action, next_action))


def samples(root, episodes=30, steps=9, conditioned=True):
    x=[]; y=[]; histories=[]
    for seed in range(root,root+episodes):
        record=episode(seed,steps)
        for t in range(1,steps-1):
            x.append(feature(record,t,conditioned))
            # Future RGB is *only* the target for the supervised fit.
            y.append(center(record["frames"][t+1])-center(record["frames"][t]))
            histories.append((seed,t))
    return np.stack(x),np.stack(y),histories


def ridge(x,y,penalty=.05):
    if penalty <= 0 or len(x)!=len(y):
        raise ValueError("Ridge needs paired samples and positive penalty")
    return np.linalg.solve(x.T@x+penalty*np.eye(x.shape[1]),x.T@y)


def evaluate(models, records, steps):
    full,targets,_=samples(91000,records,steps,True)
    blind,_,_=samples(91000,records,steps,False)
    rng=np.random.default_rng(710)
    wrong=full.copy()
    wrong[:,5:7]=full[rng.permutation(len(full)),5:7]
    observed_delta=full[:,1:3]
    reference=observed_delta+full[:,3:5]-full[:,5:7]
    predictions={
        "learned_action":full@models["action"],
        "learned_blind":blind@models["blind"],
        "shuffled_action":wrong@models["action"],
        "coasting_action":reference,
        "coasting_no_action":observed_delta}
    return {k:{"motion_mae_px":float(np.linalg.norm(v-targets,axis=-1).mean()),
               "motion_mse":float(np.square(v-targets).mean())}
            for k,v in predictions.items()}


def run(out="runs/development-28/metrics.json",train_episodes=50,
        validation_episodes=12,test_episodes=20,steps=9):
    if min(train_episodes,validation_episodes,test_episodes)<2 or steps<4:
        raise ValueError("Insufficient independent episodes")
    models={}
    for name,conditioned in (("action",True),("blind",False)):
        x,y,_=samples(31000,train_episodes,steps,conditioned)
        models[name]=ridge(x,y)
    # Independent validation produces a calibration diagnostic, not test selection.
    valid={}
    for name,conditioned in (("action",True),("blind",False)):
        vx,vy,_=samples(61000,validation_episodes,steps,conditioned)
        valid[name]=float(np.square(vx@models[name]-vy).mean())
    scores=evaluate(models,test_episodes,steps)
    report={"roadmap_id":28,"train_episodes":train_episodes,
            "validation_episodes":validation_episodes,
            "test_episodes":test_episodes,"updates":"one ridge fit per arm",
            "validation_mse":valid,"test":scores,
            "limits":"RGB-derived center and known camera commands. "
                     "A trained linear motion head, not learned raw-pixel video synthesis; "
                     "held-out camera_pan worlds only. World-state truth is unused in fit."}
    if out is not None:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/development-28/metrics.json")
    p.add_argument("--train-episodes",type=int,default=50)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.train_episodes),indent=2))


if __name__=="__main__":
    main()
