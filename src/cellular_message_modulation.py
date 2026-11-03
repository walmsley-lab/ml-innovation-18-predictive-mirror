"""Receiver-side modulation of cellular messages with matched input state."""
import jax
import jax.numpy as jnp
from src.model import conv

MODES=("plain","gated","shuffled_gate")


def init(key,channels=8):
    if channels<1:
        raise ValueError("channels must be positive")
    a,b,c,d=jax.random.split(key,4)
    inputs=channels+3  # state + signal + relevance cue + destination
    return {
        "message":.06*jax.random.normal(a,(3,3,inputs,channels)),
        "gate":.06*jax.random.normal(b,(3,3,inputs,channels)),
        "mix":.06*jax.random.normal(c,(channels,channels)),
        "readout":.03*jax.random.normal(d,(channels,1)),
    }


def step(params,hidden,signal,cue,destination,mode="gated"):
    if mode not in MODES:
        raise ValueError("Unknown message modulation mode")
    features=jnp.concatenate((hidden,signal,cue,destination),axis=-1)
    message=conv(features,params["message"])
    gate=jax.nn.sigmoid(conv(features,params["gate"]))
    if mode=="plain":
        effective=message
    elif mode=="shuffled_gate":
        effective=message*(2*jnp.roll(gate,3,axis=1))
    else:
        effective=message*(2*gate)
    return .7*hidden+.3*jnp.tanh(effective@params["mix"])


def rollout(params,pulses,cues,destination,ticks=10,mode="gated"):
    if (pulses.shape!=cues.shape or pulses.shape!=destination.shape
            or pulses.ndim!=4 or pulses.shape[-1]!=1 or ticks<1):
        raise ValueError("Need three aligned [B,H,W,1] fields")
    batch,height,width,_=pulses.shape
    initial=jnp.zeros((batch,height,width,params["readout"].shape[0]),jnp.float32)
    def update(hidden,index):
        pulse=jnp.where(index==0,pulses,jnp.zeros_like(pulses))
        hidden=step(params,hidden,pulse,cues,destination,mode)
        out=jnp.tanh(hidden@params["readout"])
        return hidden,jnp.sum(out*destination,axis=(1,2,3))
    _,signals=jax.lax.scan(update,initial,jnp.arange(ticks))
    return jnp.swapaxes(signals,0,1)  # [B,ticks]


def loss(params,pulses,cues,destination,sign,ticks=10,mode="gated"):
    predicted=rollout(params,pulses,cues,destination,ticks,mode)
    return jnp.mean((predicted[:,-min(3,ticks):]-sign[:,None])**2)
