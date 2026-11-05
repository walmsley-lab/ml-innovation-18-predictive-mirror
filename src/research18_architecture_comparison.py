"""Representation-matched RGB predictor evaluation, causal histories only."""
import numpy as np

def persistence(context,horizon):
    x=np.asarray(context,float)
    return np.repeat(x[-1:,...],horizon,axis=0)

def constant_velocity(context,horizon):
    x=np.asarray(context,float)
    if len(x)<2:raise ValueError("Need two observed frames")
    delta=x[-1]-x[-2]
    return np.stack([np.clip(x[-1]+(k+1)*delta,0,1) for k in range(horizon)])

def score(predictor,frames,context=5,horizon=3):
    x=np.asarray(frames,float)
    if context<2 or horizon<1 or len(x)<context+horizon:
        raise ValueError("Invalid held-out episode")
    forecast=np.asarray(predictor(x[:context].copy(),horizon),float)
    truth=x[context:context+horizon]
    if forecast.shape!=truth.shape:raise ValueError("Invalid prediction shape")
    if not np.isfinite(forecast).all():raise ValueError("Nonfinite forecast")
    per_lead=np.mean((forecast-truth)**2,axis=tuple(range(1,truth.ndim)))
    return {"mse":float(np.mean(per_lead)),
            "mse_by_lead":per_lead.tolist(),
            "context":int(context),"horizon":int(horizon)}

def paired_compare(models,episodes,context=5,horizon=3):
    if not episodes:raise ValueError("Need independent episodes")
    return {name:[score(fn,ep,context,horizon) for ep in episodes]
            for name,fn in models.items()}
