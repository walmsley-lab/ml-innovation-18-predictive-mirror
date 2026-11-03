"""Minimal learned cellular signal propagation, independent of RGB video."""
import jax
import jax.numpy as jnp
from src.model import conv


def init(key, channels=8):
    if channels < 1:
        raise ValueError("channels must be positive")
    a,b,c=jax.random.split(key,3)
    return {
        "perception": .05*jax.random.normal(a,(3,3,channels+2,channels)),
        "mix": .05*jax.random.normal(b,(channels,channels)),
        "readout": .03*jax.random.normal(c,(channels,1)),
    }


def step(params, hidden, pulse, destination, allow_messages=True):
    """Same cellular rule everywhere. Only tick zero receives a pulse."""
    features=jnp.concatenate((hidden,pulse,destination),axis=-1)
    kernel=params["perception"]
    if not allow_messages:
        # No spatial signal paths; local state and trainable weights remain.
        center=jnp.zeros((3,3,1,1),kernel.dtype).at[1,1].set(1)
        kernel=kernel*center
    message=conv(features,kernel)
    proposal=jnp.tanh(message@params["mix"])
    return .7*hidden+.3*proposal


def rollout(params, pulse, destination, ticks=10, allow_messages=True):
    if (pulse.shape!=destination.shape or pulse.ndim!=4 or pulse.shape[-1]!=1
            or ticks<1):
        raise ValueError("Pulse and destination must be [B,H,W,1]")
    batch,height,width,_=pulse.shape
    hidden=jnp.zeros((batch,height,width,params["readout"].shape[0]),jnp.float32)
    def tick(state,index):
        emission=jnp.where(index==0,pulse,jnp.zeros_like(pulse))
        state=step(params,state,emission,destination,allow_messages)
        decoded=jnp.tanh(state@params["readout"])
        return state,decoded
    _,output=jax.lax.scan(tick,hidden,jnp.arange(ticks))
    return output  # [ticks,B,H,W,1]


def loss(params, pulse, destination, sign, ticks=10, allow_messages=True):
    decoded=rollout(params,pulse,destination,ticks,allow_messages)
    signal=jnp.sum(decoded*destination[None],axis=(2,3,4))
    # Reward arrival *and persistence* over final three updates.
    tail=signal[-min(ticks,3):]
    return jnp.mean((tail-sign[None])**2)


def target_signal(params,pulse,destination,ticks=10,allow_messages=True):
    result=rollout(params,pulse,destination,ticks,allow_messages)
    return jnp.sum(result[-1]*destination,axis=(1,2,3))
