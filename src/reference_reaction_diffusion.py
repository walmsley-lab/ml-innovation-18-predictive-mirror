"""Reproducible Gray-Scott spatial dynamics and a shared-rule JAX NCA control.

Reference-inspired, not an exact Richardson et al. 2024 reproduction.
"""
import numpy as np
import jax
import jax.numpy as jnp
from src.model import conv


def laplace(field):
    return (-field + .25 * (np.roll(field,1,0)+np.roll(field,-1,0)
                               +np.roll(field,1,1)+np.roll(field,-1,1)))


def start(seed=11,size=24):
    if size < 12:
        raise ValueError("Size too small")
    rng=np.random.default_rng(seed)
    state=np.zeros((size,size,2),np.float32)
    state[...,0]=1
    half=max(2,size//10)
    cy=cx=size//2
    state[cy-half:cy+half,cx-half:cx+half,0]=.50
    state[cy-half:cy+half,cx-half:cx+half,1]=.25
    state += rng.normal(scale=.004,size=state.shape).astype(np.float32)
    return np.clip(state,0,1)


def step(state,feed=.035,kill=.060,du=.16,dv=.08,dt=1.0):
    if state.ndim!=3 or state.shape[-1]!=2 or not all(
            np.isfinite(z) for z in (feed,kill,du,dv,dt)) or dt<=0:
        raise ValueError("Need 2-channel finite state and positive dt")
    u,v=state[...,0],state[...,1]
    uvv=u*v*v
    un=u+dt*(du*laplace(u)-uvv+feed*(1-u))
    vn=v+dt*(dv*laplace(v)+uvv-(feed+kill)*v)
    return np.stack([un,vn],-1).astype(np.float32)


def sequence(seed=11,size=24,frames=12,burnin=8,substeps=5):
    if frames<2 or burnin<0 or substeps<1:
        raise ValueError("Invalid sequence")
    z=start(seed,size)
    for _ in range(burnin*substeps):
        z=step(z)
    seq=[]
    for _ in range(frames):
        seq.append(z)
        for _ in range(substeps):
            z=step(z)
    return np.stack(seq)


def init(key,width=24):
    if width<2:
        raise ValueError("Invalid channel width")
    k1,k2=jax.random.split(key)
    return {"local":jax.random.normal(k1,(3,3,2,width))*.015,
            "readout":jax.random.normal(k2,(width,2))*.015}


def predict_one(params,field):
    h=jnp.tanh(conv(field,params["local"]))
    return field+.1*jnp.tanh(h@params["readout"])


def rollout(params,start_field,horizon):
    if horizon<1:
        raise ValueError("Positive horizon required")
    def evolve(state, _):
        next_state=predict_one(params,state)
        return next_state,next_state
    _,preds=jax.lax.scan(evolve,start_field,None,length=horizon)
    return jnp.swapaxes(preds,0,1)
