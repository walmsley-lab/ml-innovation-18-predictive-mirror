"""Roadmap 27: distinguish observer motion from moving objects.

Action telemetry, when provided, is a permitted past/committed action. The
background registration control instead estimates global motion from observed
RGB alone; object and camera trajectories are evaluator-only.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def shift(image, dx):
    out=np.zeros_like(image)
    if dx >= 0:
        if dx < image.shape[1]: out[:,dx:] = image[:,:image.shape[1]-dx]
    else:
        if -dx < image.shape[1]: out[:,:dx] = image[:,-dx:]
    return out


def world(seed=11, steps=12, size=32):
    if steps < 4 or size < 24:
        raise ValueError("Need a stable background and four transitions")
    rng=np.random.default_rng(seed)
    background=rng.uniform(.08,.55,(size,size,3)).astype(np.float32)
    acts=rng.integers(-1,2,size=steps-1)
    offsets=np.concatenate(([0],np.cumsum(acts)))
    offsets=np.clip(offsets,-3,3).astype(int)
    actions=np.diff(offsets)
    frames=[]; objects=[]
    for t in range(steps):
        img=shift(background,-int(offsets[t]))
        x=int(9+t)-int(offsets[t])
        y=size//2
        img[y-1:y+2,x-1:x+2]=[1.0,.05,.05]
        frames.append(img)
        objects.append([9+t,y])
    return np.stack(frames), actions, np.asarray(objects,float), offsets


def detect_red(frame):
    mask=(frame[...,0]>.85)&(frame[...,1]<.2)
    xy=np.argwhere(mask)
    if not len(xy):
        raise ValueError("Missing visible target")
    return xy.mean(axis=0)[::-1]


def register_background(previous, current, radius=2):
    """Robust translation score; excludes high-error moving-object pixels."""
    costs=[]
    for d in range(-radius,radius+1):
        shifted=shift(previous,d)
        margin=radius+2
        residual=np.mean(np.abs(shifted[:,margin:-margin] -
                                current[:,margin:-margin]),axis=-1)
        costs.append(float(np.quantile(residual,.75)))
    return int(np.argmin(costs)-radius)


def world_velocity(frames, actions=None, mode="action"):
    """Recover world-space target motion from two observed images per step."""
    if mode not in ("action","no_action","image_registration"):
        raise ValueError("Bad action control")
    positions=np.stack([detect_red(x) for x in frames])
    pixel_delta=np.diff(positions[:,0])
    if mode=="action":
        if actions is None or len(actions)!=len(pixel_delta):
            raise ValueError("Known action required for the action arm")
        compensation=np.asarray(actions)
    elif mode=="image_registration":
        shifts=np.array([register_background(a,b) for a,b in zip(frames[:-1],frames[1:])])
        compensation=-shifts
    else:
        compensation=np.zeros_like(pixel_delta)
    return pixel_delta+compensation


def run(seed=11, out="runs/development-27/metrics.json"):
    frames, actions, hidden_positions, _=world(seed)
    true=np.diff(hidden_positions[:,0])
    predictions={
        "correct_action":world_velocity(frames,actions,"action"),
        "no_action":world_velocity(frames,mode="no_action"),
        "shuffled_action":world_velocity(frames,np.roll(actions,1),"action"),
        "image_registration":world_velocity(frames,mode="image_registration")}
    metrics={name:{"world_velocity_mae_px":float(np.abs(p-true).mean()),
                   "signed_bias_px":float(np.mean(p-true))}
             for name,p in predictions.items()}
    report={"roadmap_id":27,"seed":int(seed),"scores":metrics,
            "camera_actions_known":actions.tolist(),
            "limits":"Synthetic global horizontal camera shift and thresholded colored "
                     "target. Current-frame registration is a state estimator, not "
                     "future-action knowledge. No natural optical-flow validation."}
    if out is not None:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/development-27/metrics.json")
    p.add_argument("--seed",type=int,default=11)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.out),indent=2))


if __name__=="__main__":
    main()
