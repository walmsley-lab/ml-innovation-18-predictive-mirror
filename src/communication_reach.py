"""Empirical communication reach, including hidden, RGB and flow pathways.

This is a controlled finite perturbation, not an inferred causal graph.
Theoretical cellular bound k*r holds only for the specified purely local
cell rule; direct renderer/gather paths have different access geometry.
"""
import numpy as np
import jax
import jax.numpy as jnp
from src import predictive_rgb_nca as nca, rgb_motion_transport as transport
from src import rgb_multiscale_sheets as multi


def radial(response,origin,threshold=.01):
    response=np.asarray(response,np.float64)
    if response.ndim!=2 or response.size==0 or not np.isfinite(response).all():
        raise ValueError("Expected finite 2D sensitivities")
    yy,xx=np.indices(response.shape)
    d=np.sqrt((yy-origin[0])**2+(xx-origin[1])**2)
    peak=float(response.max())
    active=response>=max(1e-8,threshold*peak)
    limit=int(np.ceil(d.max()))
    return {"peak":peak,
            "measured_reach_px":float(d[active].max()) if active.any() else 0.,
            "response_by_radius":[float(response[(d>=r)&(d<r+1)].mean())
                if np.any((d>=r)&(d<r+1)) else 0.
                for r in range(limit+1)],
            "active_pixels":int(active.sum())}


def perturb_rgb(params,context,horizon=2,ticks=2,amount=.02,model="nca"):
    if context.ndim!=5 or len(context)!=1 or context.shape[-1]!=3:
        raise ValueError("One batched RGB context required")
    h,w=context.shape[2:4];cy,cx=h//2,w//2
    if model=="nca":
        predict=lambda frames:nca.rollout(params,frames,horizon,ticks)
    elif model=="fast_slow_single":
        predict=lambda frames:multi.rollout(params,frames,horizon,ticks,"single")
    elif model=="fast_slow_feedback":
        predict=lambda frames:multi.rollout(params,frames,horizon,ticks,"multiscale")
    else:
        raise ValueError("Unknown RGB pathway")
    tangent=np.zeros(context.shape,np.float32)
    tangent[0,-1,cy,cx,0]=amount
    baseline=np.asarray(predict(jnp.asarray(context)))
    modified=np.asarray(predict(jnp.asarray(context+tangent)))
    sensitivity=np.abs(modified-baseline).mean(axis=-1)[0]
    return {"path":model,"per_lead":[radial(sensitivity[t],(cy,cx))
                                     for t in range(horizon)]}


def perturb_hidden(params,size=24,ticks=4,amount=.1):
    hidden=jnp.zeros((1,size,size,params["rgb_readout"].shape[0]),jnp.float32)
    image=jnp.full((1,size,size,3),.13)
    surprise=jnp.zeros_like(image)
    tangent=hidden.at[:,size//2,size//2,0].set(amount)
    x=hidden; y=tangent
    results=[]
    for step in range(1,ticks+1):
        x=nca.cell_update(params,x,image,surprise)
        y=nca.cell_update(params,y,image,surprise)
        signal=np.asarray(jnp.mean(abs(y-x),axis=-1))[0]
        results.append({**radial(signal,(size//2,size//2)),
                        "theoretical_local_radius_px":step})
    return {"path":"isolated_cell_hidden","per_tick":results}


def perturb_flow(params,context,amount=.02):
    """Measure influence on *predicted flow weights*, not warped RGB."""
    if context.ndim!=5 or context.shape[1]<2:
        raise ValueError("Need two past RGB frames")
    h,w=context.shape[2:4];cy,cx=h//2,w//2
    previous,current=map(jnp.asarray,(context[:,-2],context[:,-1]))
    tangent=np.zeros_like(np.asarray(current))
    tangent[0,cy,cx,0]=amount
    def weighted_flow(frame):
        _,weights=transport.one_step(params,previous,frame,
                                     "transport",return_weights=True)
        coords=jnp.asarray(transport.OFFSETS,jnp.float32)
        return jnp.einsum("bhwk,kd->bhwd",weights,coords)
    baseline=np.asarray(weighted_flow(current))
    changed=np.asarray(weighted_flow(current+jnp.asarray(tangent)))
    signal=np.linalg.norm((changed-baseline)[0],axis=-1)
    return {"path":"learned_flow_router","flow_response":radial(signal,(cy,cx)),
            "flow_offset_budget_px":float(max(max(abs(y),abs(x))
                                         for y,x in transport.OFFSETS))}


def compare(seed=11,size=24,context=4,horizon=3,ticks=2,channels=6):
    if size<12 or context<2 or ticks<1 or horizon<1:
        raise ValueError("Invalid reach budget")
    rng=np.random.default_rng(int(seed))
    frames=rng.uniform(.08,.5,(1,context,size,size,3)).astype(np.float32)
    p=nca.init(jax.random.PRNGKey(int(seed)),channels)
    # Probe nonzero readout, otherwise standard zero-init has trivial RGB reach.
    p={**p,"rgb_readout":jnp.ones_like(p["rgb_readout"])*.025}
    t=transport.init(jax.random.PRNGKey(int(seed)+1),channels)
    m=multi.init(jax.random.PRNGKey(int(seed)+2),channels)
    rgb=perturb_rgb(p,frames,horizon,ticks)
    fast=perturb_rgb(m,frames,horizon,ticks,model="fast_slow_single")
    slow=perturb_rgb(m,frames,horizon,ticks,model="fast_slow_feedback")
    return {"seed":int(seed),"size":size,"context":context,"horizon":horizon,
            "cell_ticks_per_frame":ticks,
            "paths":{"nca_rgb":rgb,"cellular_hidden":perturb_hidden(
                    p,size,ticks),"learned_flow":perturb_flow(t,frames),
                    "fast_only":fast,"fast_slow":slow},
            "limits":"Finite differences on randomly initialized parameters, "
             "NOT trained dependency or attribution. Skip/renderer access may "
             "bypass local hidden-state reach; latent flow predictions and "
             "RGB synthesis must be evaluated separately."}
