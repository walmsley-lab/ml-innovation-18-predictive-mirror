"""Roadmap 36: train compact RGB latent dynamics, not a new model framework.

PCA and ridge transition learn solely from train observations. Rank and ridge
are selected on separate validation episodes. Future test frames are used only
for final metrics; a forecasting call consumes the observed prefix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src import rgb_world_ladder as worlds, rgb_benchmark as benchmark

RANKS=(2,4,8)
RIDGES=(.01,.3)


def scenes(start, count, context, horizon, size, domain="linear"):
    return np.stack([worlds.episode(start+i,context+horizon,size,
                                   domain=domain)["frames"] for i in range(count)])


def fit(frames,rank,penalty=.1):
    if frames.ndim!=5 or rank<1 or penalty<=0:
        raise ValueError("Need RGB episodes and positive latent rank")
    x=frames[:,:-1].reshape(-1,np.prod(frames.shape[2:]))
    y=frames[:,1:].reshape(-1,x.shape[1])
    if rank>min(len(x)-1,x.shape[1]):
        raise ValueError("Latent rank exceeds independent samples")
    center=x.mean(axis=0)
    _,_,vh=np.linalg.svd(x-center,full_matrices=False)
    basis=vh[:rank].T
    old=(x-center)@basis
    nxt=(y-center)@basis
    feature=np.column_stack((old,np.ones(len(old))))
    ridge=np.diag([penalty]*rank+[1e-8])
    w=np.linalg.solve(feature.T@feature+ridge,feature.T@nxt)
    matrix=w[:-1]
    # A bounded transition prevents absurdly exploding pixel forecasts.
    singular=np.linalg.svd(matrix,compute_uv=False)[0]
    matrix=matrix/max(1.,singular)
    return {"mean":center,"basis":basis,"matrix":matrix,"bias":w[-1]}


def predict(model,history,horizon):
    if history.ndim!=5 or len(history)<1 or horizon<1:
        raise ValueError("Need RGB observed prefix and positive horizon")
    center,basis=model["mean"],model["basis"]
    x=(history[:,-1].reshape(len(history),-1)-center)@basis
    output=[]
    for _ in range(horizon):
        x=x@model["matrix"]+model["bias"]
        images=np.clip(x@basis.T+center,0,1)
        output.append(images.reshape(history.shape[0],*history.shape[2:]))
    return np.stack(output,axis=1)


def validation_mse(model,frames,context,horizon):
    return float(np.square(predict(model,frames[:,:context],horizon)-
                           frames[:,context:context+horizon]).mean())


def run(out="runs/roadmap-36",train_episodes=12,valid_episodes=6,
        test_episodes=6,context=5,horizon=3,size=20):
    if min(train_episodes,valid_episodes,test_episodes)<2 or context<2 or horizon<1:
        raise ValueError("Need independent episodes and valid forecast horizon")
    train=scenes(31000,train_episodes,context,horizon,size)
    valid=scenes(61000,valid_episodes,context,horizon,size)
    candidates=[]
    for rank in RANKS:
        for penalty in RIDGES:
            fitted=fit(train,rank,penalty)
            candidates.append((validation_mse(fitted,valid,context,horizon),
                               rank,penalty,fitted))
    selected=min(candidates,key=lambda entry:(entry[0],entry[1],entry[2]))
    _,rank,penalty,model=selected
    path=Path(out) if out else None
    if path:
        path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/"latent.npz",**model)
    cases={}
    for domain in ("linear","accelerating","occlusion_long","camera_pan","combined"):
        test=scenes(91000,test_episodes,context,horizon,size,domain)
        pred=predict(model,test[:,:context],horizon)
        controls=benchmark.controls(test[:,:context],horizon)
        cases[domain]={
            "latent":benchmark.score(pred,test,context,(1,horizon)),
            **{name:benchmark.score(value,test,context,(1,horizon))
               for name,value in controls.items()}}
    report={"roadmap_id":36,"rank":rank,"ridge":penalty,
            "validation_selected_mse":selected[0],
            "train_episodes":train_episodes,"validation_episodes":valid_episodes,
            "test_episodes_per_domain":test_episodes,
            "checkpoint_sha256":(hashlib.sha256((path/"latent.npz").read_bytes()).hexdigest()
                                  if path else None),
            "cases":cases,"latent_parameter_count":int(
                sum(v.size for v in model.values())),
            "limits":"Train-only linear PCA plus stable ridge transition, not learned "
                     "nonlinear neural latent dynamics. Uses one last observed frame, "
                     "while constant-velocity baseline uses two; record unequal "
                     "history capacity. No architectural win claimed."}
    if path:
        (path/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-36")
    p.add_argument("--train-episodes",type=int,default=12)
    a=p.parse_args()
    print(json.dumps(run(a.out,train_episodes=a.train_episodes)["cases"],indent=2))


if __name__=="__main__":
    main()
