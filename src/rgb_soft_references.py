"""Differentiable content-conditioned offset selection for RGB NCA cells.

Every pixel computes soft weights over a small fixed set of offset candidates
using ONLY its own observed/predicted state. This learns *routing weights*,
not arbitrary object references, correspondences, or identity.
"""
import jax
import jax.numpy as jnp

MODES=("uniform","learned","shuffled")
OFFSETS=((0,0),(-3,0),(3,0),(0,-3),(0,3),(-6,0),(6,0),(0,-6),(0,6))


def init(key, channels=8):
    a,b=jax.random.split(key)
    return {
        "update":.05*jax.random.normal(a,(2*(channels+6),2*channels)),
        "bias":jnp.zeros((2*channels,),jnp.float32),
        "router":.05*jax.random.normal(b,(channels+6,len(OFFSETS))),
        "readout":jnp.zeros((channels,3),jnp.float32),
        "rgb_bias":jnp.zeros((3,),jnp.float32)}


def candidates(features):
    """Nonwrapping spatial offsets; fixed candidates, learned selection."""
    batch,height,width,channels=features.shape
    pad=max(max(abs(a),abs(b)) for a,b in OFFSETS)
    padded=jnp.pad(features,((0,0),(pad,pad),(pad,pad),(0,0)))
    return jnp.stack([padded[:,pad+dy:pad+dy+height,pad+dx:pad+dx+width,:]
                      for dy,dx in OFFSETS],axis=-2)


def choose(params,features,mode):
    if mode not in MODES:
        raise ValueError("Unknown reference-selection arm")
    choices=candidates(features)
    logits=features@params["router"]
    if mode=="uniform":
        weights=jnp.ones_like(logits)/len(OFFSETS)
    else:
        weights=jax.nn.softmax(logits,axis=-1)
        if mode=="shuffled":
            # Break learned offset correspondence without altering weight set.
            weights=jnp.roll(weights,3,axis=-1)
    return jnp.sum(choices*weights[...,None],axis=-2), weights


def update(params,hidden,image,error,mode):
    features=jnp.concatenate((hidden,image,error),axis=-1)
    chosen,_=choose(params,features,mode)
    gate,proposal=jnp.split(
        jnp.concatenate((features,chosen),axis=-1)@params["update"]
        +params["bias"],2,axis=-1)
    return hidden+.125*jax.nn.sigmoid(gate)*jnp.tanh(proposal)


def rollout(params,context,horizon=10,ticks=2,mode="learned",
            return_routing=False):
    if (mode not in MODES or context.ndim!=5 or context.shape[-1]!=3
            or context.shape[1]<2 or horizon<1 or ticks<1):
        raise ValueError("Need RGB [B,C,H,W,3] and valid horizon/ticks/arm")
    batch,_,height,width,_=context.shape
    hidden=jnp.zeros((batch,height,width,params["readout"].shape[0]))
    image=jnp.zeros_like(context[:,0])
    state=(hidden,image)

    def cellular(h,frame,error):
        def tick(carry,_):
            return update(params,carry,frame,error,mode),None
        carry,_=jax.lax.scan(tick,h,None,length=ticks)
        return carry

    def dream(current):
        h,frame=current
        h=cellular(h,frame,jnp.zeros_like(frame))
        correction=.2*jnp.tanh(h@params["readout"]+params["rgb_bias"])
        future=jnp.clip(frame+correction,0,1)
        return (h,future),future

    def observed_then_dream(current,frame):
        h,previous=current
        current=(cellular(h,frame,frame-previous),frame)
        current,_=dream(current)
        return current,None

    state,_=jax.lax.scan(observed_then_dream,state,
                         jnp.swapaxes(context[:,:-1],0,1))
    last=context[:,-1]
    state=(cellular(state[0],last,last-state[1]),last)

    def forecast(current,_):
        return dream(current)
    _,frames=jax.lax.scan(forecast,state,None,length=horizon)
    frames=jnp.swapaxes(frames,0,1)
    if return_routing:
        features=jnp.concatenate((state[0],last,jnp.zeros_like(last)),axis=-1)
        _,weights=choose(params,features,mode)
        return frames,weights
    return frames


def loss(params,examples,context,horizon,ticks,mode):
    prediction=rollout(params,examples[:,:context],horizon,ticks,mode)
    truth=examples[:,context:context+horizon]
    motion=jnp.mean(jnp.abs(truth-examples[:,context-1:context]),
                    axis=-1,keepdims=True)
    return jnp.mean((1+8*motion)*jnp.square(prediction-truth))
