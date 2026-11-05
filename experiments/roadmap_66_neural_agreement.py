"""66: optional MEASURED neural-response prediction with session holdouts.

No neural recordings ship with this repository. Synthetic mode checks data,
time-lag, trial/session split and regression procedures. Measured claims require
operator-supplied recordings with immutable source provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

LAGS=(1,2,3)


def toy(seed=11,sessions=12,times=20,features=3,neurons=6):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(sessions,times,features))
    weights=rng.normal(size=(features,neurons))
    y=rng.normal(scale=.1,size=(sessions,times,neurons))
    y[:,2:]+=x[:,:-2]@weights
    return x,y,np.arange(sessions)


def load(path,metadata):
    p=Path(path)
    m=json.loads(Path(metadata).read_text())
    if m.get("source_type")!="measured" or not all(m.get(k) for k in
            ("citation","license","species","recording_version","sha256")):
        raise ValueError("Measured neural data need citation/license/version")
    if hashlib.sha256(p.read_bytes()).hexdigest()!=m["sha256"]:
        raise ValueError("Neural recording SHA mismatch")
    with np.load(p,allow_pickle=False) as data:
        x=np.asarray(data["stimulus_features"],float)
        y=np.asarray(data["neural_response"],float)
        sessions=np.asarray(data["session_ids"])
    validate(x,y,sessions)
    return x,y,sessions,m


def validate(x,y,sessions):
    if x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=y.shape[:2] or (
            len(sessions)!=len(x)) or len(np.unique(sessions))<5 or
            x.shape[1]<6 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Need aligned finite [trials,time,channels] and >=5 sessions")


def splits(sessions,seed=42):
    groups=np.unique(sessions)
    if len(groups)<5:
        raise ValueError("Five independent recording groups required")
    group_order=np.random.default_rng(seed).permutation(groups)
    a=max(1,int(len(groups)*.6))
    b=max(a+1,int(len(groups)*.8))
    return [np.flatnonzero(np.isin(sessions,subset))
            for subset in (group_order[:a],group_order[a:b],group_order[b:])]


def examples(x,y,ids,lag):
    if lag not in LAGS:
        raise ValueError("Only positive causal lags permitted")
    return x[ids,:-lag].reshape(-1,x.shape[-1]),y[ids,lag:].reshape(-1,y.shape[-1])


def fit(x,y,penalty=.1):
    if x.shape[0]!=y.shape[0] or penalty<=0:
        raise ValueError("Need paired train examples and positive ridge")
    return np.linalg.solve(x.T@x+penalty*np.eye(x.shape[-1]),x.T@y)


def score(pred,target):
    mse=float(np.square(pred-target).mean())
    p=pred-pred.mean(axis=0)
    t=target-target.mean(axis=0)
    cor=np.sum(p*t,axis=0)/np.maximum(
        np.sqrt((p*p).sum(0)*(t*t).sum(0)),1e-8)
    return {"mse":mse,"neuron_pearson_mean":float(np.mean(cor))}


def run(out="runs/roadmap-66/metrics.json",path=None,metadata=None,split_seed=42):
    if bool(path)!=bool(metadata):
        raise ValueError("Recording and metadata must be paired")
    if path:
        x,y,sessions,m=load(path,metadata)
    else:
        x,y,sessions=toy()
        m={"source_type":"synthetic","citation":"self-generated latency reference"}
    validate(x,y,sessions)
    train,val,test=splits(sessions,split_seed)
    candidate=[]
    for lag in LAGS:
        a,b=examples(x,y,train,lag)
        parameters=fit(a,b)
        vx,vy=examples(x,y,val,lag)
        candidate.append((score(vx@parameters,vy)["mse"],lag,parameters))
    selected=min(candidate,key=lambda x:(x[0],x[1]))
    _,lag,coefficients=selected
    tx,ty=examples(x,y,test,lag)
    ax,ay=examples(x,y,train,lag)
    # Source-shuffled-label control with equal parameter shape.
    rng=np.random.default_rng(split_seed+999)
    control=fit(ax,ay[rng.permutation(len(ay))])
    report={"roadmap_id":66,"source_type":m["source_type"],
            "selected_lag_frames":lag,"validation_mse":selected[0],
            "test":{ "stimulus_regression":score(tx@coefficients,ty),
                     "shuffled_training_labels":score(tx@control,ty),
                     "constant_train_mean":score(
                        np.broadcast_to(ay.mean(0),ty.shape),ty)},
            "split_trials":{"training":len(train),"validation":len(val),"test":len(test)},
            "independent_session_groups":len(np.unique(sessions)),
            "noise_ceiling":None,
            "limits":"Linear stimulus-to-neural-response decoding, NOT connectome "
             "constrained dynamics. Synthetic data do not establish any "
             "neural agreement. Measured metadata are operator-supplied, not "
             "independently authenticated. Noise ceiling unavailable unless "
             "independent repeated-trial data supplied. All lags causal."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--recording");p.add_argument("--metadata")
    p.add_argument("--out",default="runs/roadmap-66/metrics.json")
    a=p.parse_args();print(json.dumps(run(a.out,a.recording,a.metadata),indent=2))
