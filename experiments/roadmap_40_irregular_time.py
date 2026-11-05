"""Roadmap 40: learn timing-aware visual motion from observed RGB.

The duration of the *scheduled* next interval is known at forecast issuance.
The actual next frame, target location and future capture timestamp are not.
Contrast time-normalized motion with equal-dimensional time-blind regression,
frame-index velocity and persistence on withheld capture cadences.
"""
import argparse
import json
from pathlib import Path
import numpy as np

DOMAINS=("regular","jitter","dropped","acceleration")


def episode(seed,steps=14,size=32,domain="regular"):
    if steps<5 or size<20 or domain not in DOMAINS:
        raise ValueError("Invalid time world")
    rng=np.random.default_rng(int(seed))
    if domain=="regular": intervals=np.full(steps-1,.14)
    else:
        intervals=rng.uniform(.075,.24,steps-1)
        if domain=="dropped":
            intervals[::4]*=2.3
    stamps=np.r_[0.,np.cumsum(intervals)]
    start=np.array([size*.4,size*.55])
    velocity=rng.uniform(1.0,2.8,2)*rng.choice([-1,1],size=2)
    accel=rng.uniform(-.6,.6,2) if domain=="acceleration" else np.zeros(2)
    positions=start+stamps[:,None]*velocity+.5*stamps[:,None]**2*accel
    yy,xx=np.mgrid[:size,:size]
    frames=[]
    for x,y in positions:
        image=np.full((size,size,3),.025,np.float32)
        blob=np.exp(-.5*((xx-x)**2+(yy-y)**2)/1.2**2)
        image[...,0]=.025+.9*blob
        image[...,1]=.025+.15*blob
        frames.append(image)
    return {"frames":np.stack(frames),"scheduled_interval":intervals,
            "timestamps":stamps,"positions_evaluation_only":positions}


def observed_center(image):
    salience=np.maximum(image[...,0]-image[...,1]-.001,0.)
    if not salience.sum()>1e-6:
        raise ValueError("Missing observed object")
    yy,xx=np.indices(salience.shape)
    return np.array([np.sum(salience*xx),np.sum(salience*yy)])/salience.sum()


def feature(record,index,time_aware=True):
    """Forecast at observation index t, with scheduled interval t->t+1."""
    frames=record["frames"]
    if not 1<=index<len(frames)-1:
        raise ValueError("Need two earlier frames and one future target")
    prev,current=map(observed_center,frames[index-1:index+1])
    duration=record["scheduled_interval"]
    ratio=duration[index]/duration[index-1] if time_aware else 0.
    observed_delta=current-prev
    return np.array([1.,*observed_delta,*(observed_delta*ratio)]),current


def design(start,count,domain,steps=14):
    xs=[];ys=[];groups=[];baseline=[]
    for seed in range(start,start+count):
        r=episode(seed,steps,domain=domain)
        for t in range(1,steps-1):
            x,current=feature(r,t)
            target=observed_center(r["frames"][t+1])-current
            xs.append(x);ys.append(target);groups.append(seed)
            blind,_=feature(r,t,False)
            baseline.append(blind)
    return np.asarray(xs),np.asarray(ys),np.asarray(baseline),np.asarray(groups)


def fit(features,targets,penalty=.05):
    if penalty<=0 or len(features)!=len(targets):
        raise ValueError("Matched positive-ridge fit required")
    return np.linalg.solve(features.T@features+
                           penalty*np.eye(features.shape[1]),
                           features.T@targets)


def measures(predicted,target):
    error=np.linalg.norm(predicted-target,axis=-1)
    return {"displacement_mae_px":float(error.mean()),
            "displacement_mse_px2":float(np.mean(error**2))}


def run(out="runs/roadmap-40/metrics.json",train_count=30,
        validation_count=10,test_count=15):
    if min(train_count,validation_count,test_count)<2:
        raise ValueError("Need disjoint independent whole episodes")
    train=design(31000,train_count,"jitter")
    validation=design(61000,validation_count,"jitter")
    model={}
    validation_scores={}
    for mode in ("time_aware","frame_blind"):
        tx,ty,tblind,_=train
        vx,vy,vblind,_=validation
        train_x=tx if mode=="time_aware" else tblind
        val_x=vx if mode=="time_aware" else vblind
        trials=[]
        for penalty in (.01,.1,1.):
            coefficients=fit(train_x,ty,penalty)
            trials.append((measures(val_x@coefficients,vy)["displacement_mse_px2"],
                           penalty,coefficients))
        best=min(trials,key=lambda x:(x[0],x[1]))
        validation_scores[mode]=best[0]
        model[mode]=best[2]
    results={}
    for domain in DOMAINS:
        x,y,blind,_=design(91000,test_count,domain)
        forecast={"time_aware":x@model["time_aware"],
                  "frame_blind":blind@model["frame_blind"],
                  "scheduled_velocity":x[:,3:5],
                  "per_frame_velocity":x[:,1:3],
                  "persistence":np.zeros_like(y)}
        results[domain]={k:measures(v,y) for k,v in forecast.items()}
    report={"roadmap_id":40,"train_episodes":train_count,
            "validation_episodes":validation_count,"test_episodes_per_domain":test_count,
            "fitted_validation_mse":validation_scores,"test":results,
            "limits":"RGB centroid observations and *planned* future sample interval. "
             "Future actual capture time is unknown to these predictors. "
             "Simple ridge one-step dynamics, not continuous-time recurrent NCA. "
             "Time-aware and blind arms have equal coefficient matrices but "
             "differ in active features; no performance claim."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-40/metrics.json")
    a=p.parse_args()
    print(json.dumps(run(a.out)["test"],indent=2))


if __name__=="__main__":
    main()
