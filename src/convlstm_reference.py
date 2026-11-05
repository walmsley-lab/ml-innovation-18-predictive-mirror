"""Minimal single-layer JAX ConvLSTM visual-prediction reference.

ConvLSTM is established prior art (Shi et al. 2015); this is a small,
transparent local baseline, not a faithful reproduction of a full paper model.
"""
import jax
import jax.numpy as jnp
from src.model import conv


def init(key,width=8):
    if width<2:
        raise ValueError("Need at least two hidden channels")
    a,b=jax.random.split(key)
    return {
        "gates":jax.random.normal(a,(3,3,width+3,4*width))*.05,
        "bias":jnp.zeros((4*width,),jnp.float32),
        "rgb":jax.random.normal(b,(width,3))*.04,
        "rgb_bias":jnp.full((3,),-2.,jnp.float32)
    }


def initial(params,batch,height,width):
    c=params["rgb"].shape[0]
    zeros=jnp.zeros((batch,height,width,c),jnp.float32)
    return zeros,zeros


def update(params,state,frame):
    h,c=state
    z=conv(jnp.concatenate([frame,h],-1),params["gates"])+params["bias"]
    i,f,g,o=jnp.split(z,4,-1)
    # Positive forget bias makes temporal memory easier at initialization.
    i=jax.nn.sigmoid(i)
    f=jax.nn.sigmoid(f+1.)
    o=jax.nn.sigmoid(o)
    c=f*c+i*jnp.tanh(g)
    h=o*jnp.tanh(c)
    return h,c


def rollout(params,context,horizon):
    if context.ndim!=5 or context.shape[-1]!=3 or context.shape[1]<2 or horizon<1:
        raise ValueError("Need [B,T,H,W,3], T>=2 and positive horizon")
    b,t,ht,w,_=context.shape
    state=initial(params,b,ht,w)
    def absorb(s,frame):
        return update(params,s,frame),None
    state,_=jax.lax.scan(absorb,state,jnp.swapaxes(context,0,1))
    def dream(carry,_):
        state,prev=carry
        state=update(params,state,prev)
        candidate=jax.nn.sigmoid(state[0]@params["rgb"]+params["rgb_bias"])
        return (state,candidate),candidate
    _,forecast=jax.lax.scan(dream,(state,context[:,-1]),None,length=horizon)
    return jnp.swapaxes(forecast,0,1)


def loss(params,examples,context=8,horizon=10,foreground_weight=4):
    prediction=rollout(params,examples[:,:context],horizon)
    truth=examples[:,context:context+horizon]
    foreground=(jnp.max(truth,-1,keepdims=True)>.20).astype(jnp.float32)
    return jnp.mean((1+foreground_weight*foreground)*(prediction-truth)**2)
