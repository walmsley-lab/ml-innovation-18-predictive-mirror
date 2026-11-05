"""Roadmap 34: causal visual-motion and inertial signal fusion.

The visual proxy is extracted from rendered RGB; the second modality is a
noisy camera-velocity sensor already measured at forecast issuance. No future
camera motion, world position, or simulator velocity enters model features.
"""
import argparse
import json
from pathlib import Path
import numpy as np

WIDTH = 40


def episode(seed, steps=20, width=WIDTH):
    if steps < 5 or width < 28:
        raise ValueError("Need a visible moving target")
    rng = np.random.default_rng(int(seed))
    object_speed = float(rng.uniform(-.24, .24))
    cam_speed = np.zeros(steps,float)
    camera = np.zeros(steps,float)
    for t in range(1,steps):
        cam_speed[t] = .86*cam_speed[t-1] + rng.normal(0,.17)
        camera[t] = camera[t-1] + cam_speed[t]
    center = .5*width + object_speed*np.arange(steps) - camera
    pixels = np.arange(width,dtype=float)
    frames=np.zeros((steps,1,width,3),np.float32)
    for t,x in enumerate(center):
        frames[t,0,:,0] = np.exp(-.5*((pixels-x)/.85)**2)
    imu = cam_speed + rng.normal(0,.055,steps)
    return {"frames":frames,"imu":imu.astype(np.float32),
            "evaluation_center":center, "evaluation_camera":camera}


def visual_center(image):
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Need one RGB observation")
    intensity=np.maximum(np.asarray(image)[...,0]-.075,0.).sum(axis=0)
    if float(intensity.sum()) < 1e-7:
        raise ValueError("No visible object")
    return float(np.dot(intensity,np.arange(len(intensity))) / intensity.sum())


def features(frames, imu, t, mode="fused", missing=False):
    """Predict movement from observation t to t+1, before receiving t+1."""
    if mode not in ("fused","vision","inertial") or not 1<=t<len(frames)-1:
        raise ValueError("Invalid causal forecast")
    if len(imu) != len(frames):
        raise ValueError("Sensor timestamps must align with video")
    velocity=(visual_center(frames[t])-visual_center(frames[t-1])
              if mode!="inertial" else 0.)
    sensor=(float(imu[t]) if mode!="vision" and not missing else 0.)
    flag=float(mode=="vision" or missing)
    return np.array([1., velocity, sensor, flag],float)


def records(first_seed,count=30,steps=20):
    if count < 1:
        raise ValueError("Need independent full episodes")
    xx, yy, group=[],[],[]
    for seed in range(first_seed,first_seed+count):
        r=episode(seed,steps)
        centers=[visual_center(frame) for frame in r["frames"]]
        for t in range(1,steps-1):
            xx.append(features(r["frames"],r["imu"],t))
            yy.append(centers[t+1]-centers[t])
            group.append(int(seed))
    return np.asarray(xx),np.asarray(yy),np.asarray(group)


def mask_features(x,mode,drop=None):
    out=np.asarray(x,float).copy()
    if mode=="vision":
        out[:,2]=0.
        out[:,3]=1.
    elif mode=="inertial":
        out[:,1]=0.
        out[:,3]=0.
    elif mode=="fused":
        out[:,3]=0.
    elif mode=="robust":
        if drop is None or len(drop)!=len(out):
            raise ValueError("Need observed dropout flags")
        out[drop,2]=0.
        out[:,3]=np.asarray(drop,float)
    else:
        raise ValueError("Unknown modality mode")
    return out


def fit(x,y,penalty=.1):
    if x.shape[0]!=len(y) or penalty<=0:
        raise ValueError("Bad matched training")
    return np.linalg.solve(x.T@x+penalty*np.eye(x.shape[1]),x.T@y)


def scores(weights,x,y,missing):
    variants={
        "full":mask_features(x,"fused"),
        "sensor_dropped":mask_features(x,"robust",missing),
        "inertial_shuffled":mask_features(x,"fused")}
    rng=np.random.default_rng(77193)
    variants["inertial_shuffled"][:,2]=x[rng.permutation(len(x)),2]
    report={}
    for stress,base in variants.items():
        rows={}
        for name,w in weights.items():
            observed=(mask_features(base,"vision") if name=="vision" else
                      mask_features(base,"inertial") if name=="inertial" else
                      base)
            pred=observed@w
            rows[name]={"motion_mse":float(np.square(pred-y).mean()),
                        "motion_mae":float(np.abs(pred-y).mean())}
        # Non-learned coasting and zero-motion controls.
        rows["video_coast"]={"motion_mse":float(np.square(x[:,1]-y).mean()),
                             "motion_mae":float(np.abs(x[:,1]-y).mean())}
        rows["zero_motion"]={"motion_mse":float(np.square(y).mean()),
                             "motion_mae":float(np.abs(y).mean())}
        report[stress]=rows
    return report


def run(out="runs/roadmap-34/metrics.json",train_count=45,val_count=12,test_count=20):
    if min(train_count,val_count,test_count)<2:
        raise ValueError("Need distinct train/validation/test episodes")
    tx,ty,_=records(31000,train_count)
    vx,vy,_=records(61000,val_count)
    ex,ey,groups=records(91000,test_count)
    train_drop=np.random.default_rng(113).random(len(tx))<.3
    test_drop=np.random.default_rng(223).random(len(ex))<.3
    weights={
        mode:fit(mask_features(tx,mode,train_drop if mode=="robust" else None),ty)
        for mode in ("vision","inertial","fused","robust")}
    validation={mode:float(np.square(
        mask_features(vx,"vision" if mode=="vision" else (
            "inertial" if mode=="inertial" else "fused"))@w-vy).mean())
                for mode,w in weights.items()}
    report={"roadmap_id":34,
            "train_episodes":train_count,"validation_episodes":val_count,
            "test_episodes":test_count,
            "heldout_test_seed_start":int(groups.min()),
            "training_missing_fraction":float(train_drop.mean()),
            "test_missing_fraction":float(test_drop.mean()),
            "ridge_coefficient_count_per_arm":4,
            "validation_full_sensor_mse":validation,
            "test":scores(weights,ex,ey,test_drop),
            "limits":"Observed RGB-derived 1D target location and current noisy inertial "
                     "measurement; camera acceleration is never supplied. "
                     "Linear sensor fusion, not neural cross-modal cortex. "
                     "Frozen single-domain toy camera; reported sensor dropouts "
                     "are synthetic. Equal coefficient arrays but different "
                     "effective active feature counts."}
    if out is not None:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-34/metrics.json")
    a=p.parse_args()
    print(json.dumps(run(a.out)["test"],indent=2))


if __name__=="__main__":
    main()
