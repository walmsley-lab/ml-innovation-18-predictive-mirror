"""A tiny two-frame convolutional RGB forecaster; no NCA recurrent state.

A deliberately boring learned baseline for checking whether cellular
complexity provides value. Context enters only through its observed prefix.
"""
import jax
import jax.numpy as jnp
from src.model import conv


def init(key, width=16):
    if width < 1:
        raise ValueError("width must be positive")
    a,b,c = jax.random.split(key,3)
    return {
        "conv1": .06*jax.random.normal(a,(3,3,6,width)),
        "conv2": .06*jax.random.normal(b,(3,3,width,width)),
        "readout": .02*jax.random.normal(c,(1,1,width,3)),
        "bias": jnp.zeros((3,),jnp.float32),
    }


def one_step(params, previous, current):
    motion = current-previous
    features = jnp.concatenate((current,motion),axis=-1)
    hidden = jax.nn.relu(conv(features,params["conv1"]))
    hidden = jax.nn.relu(conv(hidden,params["conv2"]))
    residual = .30*jnp.tanh(conv(hidden,params["readout"])+params["bias"])
    return jnp.clip(current+residual,0,1)


def rollout(params, context, horizon=10):
    if (context.ndim != 5 or context.shape[-1] != 3 or
            context.shape[1] < 2 or horizon < 1):
        raise ValueError("Expected RGB [B,C,H,W,3], C>=2, positive horizon")
    def predict(carry,_):
        before,current=carry
        future=one_step(params,before,current)
        return (current,future),future
    _,result=jax.lax.scan(predict,(context[:,-2],context[:,-1]),None,length=horizon)
    return jnp.swapaxes(result,0,1)


def loss(params, examples, context=8, horizon=10):
    forecasts=rollout(params,examples[:,:context],horizon)
    return jnp.mean(jnp.square(forecasts-examples[:,context:context+horizon]))
