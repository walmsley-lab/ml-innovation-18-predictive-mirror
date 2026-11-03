"""Recurrent RGB NCA combined with learned local motion transport.

Unlike the earlier feedforward transport pilot, hidden state persists
through observed-context correction and open-loop future generation.
"""
import jax
import jax.numpy as jnp
from src import predictive_rgb_nca as nca, rgb_motion_transport as flow

MODES=("nca","transport","combined")


def init(key, channels=8):
    a,b=jax.random.split(key)
    return {"nca":nca.init(a,channels),"transport":flow.init(b,channels)}


def dream(params,state,before,current,mode,ticks=2):
    if mode not in MODES:
        raise ValueError("Unknown recurrent transport arm")
    next_nca,rgb=nca.dream(params["nca"],state,ticks)
    shifted=flow.one_step(params["transport"],before,current,"transport")
    if mode=="nca":
        future=rgb
    elif mode=="transport":
        future=shifted
    else:
        future=jnp.clip(shifted+(rgb-current),0,1)
    # The state used for the next prediction/error correction carries
    # the *committed* chosen forecast, never an alternate branch forecast.
    return (next_nca[0],future),future


def rollout(params,context,horizon=10,mode="combined",ticks=2):
    if (mode not in MODES or context.ndim!=5 or context.shape[-1]!=3
            or context.shape[1]<2 or horizon<1 or ticks<1):
        raise ValueError("Need valid RGB context and horizon")
    batch,_,height,width,_=context.shape
    first=context[:,0]
    state=nca.observe(params["nca"],
                      nca.initial_state(params["nca"],batch,height,width),
                      first,ticks)
    def absorb(carry,observed):
        state,before,current=carry
        state,_=dream(params,state,before,current,mode,ticks)
        state=nca.observe(params["nca"],state,observed,ticks)
        return (state,current,observed),None
    (state,before,current),_=jax.lax.scan(
        absorb,(state,first,first),jnp.swapaxes(context[:,1:],0,1))
    def predict(carry,_):
        state,before,current=carry
        state,future=dream(params,state,before,current,mode,ticks)
        return (state,current,future),future
    _,predicted=jax.lax.scan(predict,(state,before,current),None,length=horizon)
    return jnp.swapaxes(predicted,0,1)


def loss(params,episodes,context=8,horizon=10,mode="combined",ticks=2):
    predicted=rollout(params,episodes[:,:context],horizon,mode,ticks)
    truth=episodes[:,context:context+horizon]
    motion=jnp.mean(jnp.abs(truth-episodes[:,context-1:context]),axis=-1,keepdims=True)
    return jnp.mean((1+8*motion)*jnp.square(predicted-truth))
