"""Full-state field prediction: common information for all learned arms."""
import jax
import jax.numpy as jnp
import numpy as np
from src.model import conv
from src import shared_cellular_core, physical_fields as physics

ARMS=("cellular","cnn")


def init(key,arm="cellular",channels=8):
    if arm not in ARMS or channels<1:
        raise ValueError("Invalid arm/channels")
    a,b,c=jax.random.split(key,3)
    if arm=="cellular":
        return {"core":shared_cellular_core.init(a,channels),
                "head":.04*jax.random.normal(b,(channels,2)),
                "bias":jnp.zeros((2,),jnp.float32)}
    return {"encoder":.06*jax.random.normal(a,(3,3,2,channels)),
            "hidden":.06*jax.random.normal(b,(3,3,channels,channels)),
            "head":.04*jax.random.normal(c,(channels,2)),
            "bias":jnp.zeros((2,),jnp.float32)}


def step(params,state,arm,kind):
    if arm not in ARMS or kind not in physics.KINDS or state.ndim!=4 or state.shape[-1]!=2:
        raise ValueError("Expected [B,H,W,2] with named arm/kind")
    if arm=="cellular":
        hidden=jnp.zeros((*state.shape[:-1],params["head"].shape[0]),state.dtype)
        hidden=shared_cellular_core.tick(params["core"],hidden,state,"signal")
        hidden=shared_cellular_core.tick(params["core"],hidden,state,"signal")
    else:
        hidden=jnp.tanh(conv(state,params["encoder"]))
        hidden=jnp.tanh(conv(hidden,params["hidden"]))
    next_state=state+.08*jnp.tanh(hidden@params["head"]+params["bias"])
    if kind=="diffusion":
        next_state=next_state.at[...,1].set(0.)
    return next_state


def rollout(params,state,horizon,arm,kind):
    if horizon<1:
        raise ValueError("Positive horizon required")
    def advance(x,_):
        x=step(params,x,arm,kind)
        return x,x
    _,frames=jax.lax.scan(advance,state,None,length=horizon)
    return jnp.swapaxes(frames,0,1)


def dataset(seeds,kind,context=3,horizon=5,size=20):
    records=[physics.episode(int(seed),kind,size,context+horizon) for seed in seeds]
    frames=np.stack([r["states"] for r in records])
    if kind=="diffusion":
        frames=np.concatenate([frames,np.zeros_like(frames)],axis=2)
    frames=np.moveaxis(frames,2,-1).astype(np.float32)
    return frames[:,:context],frames[:,context:],records


def fit_stencil(observed,next_state,ridge=1e-6):
    """Train-only least squares [u,v,Lap(u),Lap(v),1]->next-state delta."""
    x=np.asarray(observed,np.float64)
    y=np.asarray(next_state,np.float64)
    if x.ndim!=4 or y.shape!=x.shape or x.shape[-1]!=2:
        raise ValueError("Expected matching [N,H,W,2] fields")
    lap=np.asarray([np.stack([physics.laplacian(im[...,c])
                              for c in range(2)],-1) for im in x])
    features=np.concatenate([x,lap,np.ones((*x.shape[:3],1))],-1).reshape(-1,5)
    targets=(y-x).reshape(-1,2)
    weights=np.linalg.solve(features.T@features+ridge*np.eye(5),
                            features.T@targets)
    return weights.astype(np.float32)


def stencil_rollout(weights,initial,horizon,kind):
    x=np.asarray(initial,np.float64).copy()
    frames=[]
    for _ in range(horizon):
        lap=np.asarray([np.stack([physics.laplacian(im[...,c])
                                  for c in range(2)],-1) for im in x])
        features=np.concatenate([x,lap,np.ones((*x.shape[:3],1))],-1)
        x=x+features@weights
        if kind=="diffusion":
            x[...,1]=0
        frames.append(x.copy())
    return np.stack(frames,axis=1).astype(np.float32)


def numerical(records,context,horizon):
    rows=[]
    for r in records:
        x=r["states"][context-1].copy()
        sequence=[]
        for _ in range(horizon):
            x=physics.step(x,**r["parameters"])
            sequence.append(np.moveaxis(
                np.concatenate([x,np.zeros_like(x)],axis=0)
                if r["kind"]=="diffusion" else x,0,-1))
        rows.append(sequence)
    return np.asarray(rows,np.float32)


def metrics(pred,true,initial,kind):
    pred=np.asarray(pred)
    if pred.shape!=true.shape or not np.isfinite(pred).all():
        return {"finite":False,"stable":False,"per_lead_rmse":None}
    error=np.sqrt(np.mean((pred-true)**2,axis=(0,2,3,4)))
    amplitudes=np.max(abs(pred),axis=(2,3,4))/np.maximum(
        np.max(abs(true),axis=(2,3,4)),1e-8)
    mass=np.sum(pred[...,0],axis=(2,3))
    initial_mass=np.sum(initial[...,0],axis=(1,2))
    return {"finite":True,"per_lead_rmse":error.tolist(),
            "mean_rmse":float(np.mean(error)),"max_amplitude_ratio":float(amplitudes.max()),
            "stable":bool(np.all(amplitudes<10)),
            "diffusion_mass_drift_by_lead":np.mean(abs(mass-initial_mass[:,None]),axis=0).tolist()
                if kind=="diffusion" else None}
