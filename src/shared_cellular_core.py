"""One *exactly shared* local transition in signal and RGB domains.

No semantic labels or external positions are present in the RGB forecast.
All domain-specific sensing is in separate adapters.
"""
import jax
import jax.numpy as jnp
from src.model import conv


def init(key, channels=8):
    if channels < 1:
        raise ValueError("Need positive channels")
    a, b, c, d, e = jax.random.split(key, 5)
    return {
        "core": .08*jax.random.normal(a, (3, 3, channels, channels)),
        "signal_input": .08*jax.random.normal(b, (3, 3, 2, channels)),
        "rgb_input": .08*jax.random.normal(c, (3, 3, 6, channels)),
        "signal_readout": .06*jax.random.normal(d, (channels, 1)),
        "rgb_readout": .03*jax.random.normal(e, (channels, 3)),
    }


def tick(params, hidden, sensors, domain):
    """Identical hidden-to-hidden convolution for either domain."""
    if domain not in ("signal", "rgb"):
        raise ValueError("Domain must be signal or rgb")
    proposed = conv(hidden, params["core"]) + conv(
        sensors, params[domain + "_input"])
    return .75*hidden+.25*jnp.tanh(proposed)


def signal(params, pulse, target, ticks=10):
    if (pulse.ndim != 4 or pulse.shape != target.shape
            or pulse.shape[-1] != 1 or ticks < 1):
        raise ValueError("Invalid signal fields")
    b, h, w, _ = pulse.shape
    hidden = jnp.zeros((b, h, w, params["core"].shape[-1]))
    def step(current, index):
        injection = jnp.where(index == 0, pulse, jnp.zeros_like(pulse))
        updated = tick(params, current, jnp.concatenate((injection,target),-1), "signal")
        prediction = jnp.sum(jnp.tanh(updated @ params["signal_readout"])*target,(1,2,3))
        return updated, prediction
    _, predictions = jax.lax.scan(step, hidden, jnp.arange(ticks))
    return jnp.swapaxes(predictions,0,1)


def rgb(params, context, horizon=5):
    if (context.ndim != 5 or context.shape[-1] != 3
            or context.shape[1] < 2 or horizon < 1):
        raise ValueError("Need context [B,C,H,W,3] with C>=2")
    b, _, h, w, _ = context.shape
    hidden = jnp.zeros((b,h,w,params["core"].shape[-1]))
    zero = jnp.zeros_like(context[:,0])
    def observe(state, frame):
        hidden, previous = state
        corrected = tick(params, hidden, jnp.concatenate((frame,frame-previous),-1), "rgb")
        predicted = jnp.clip(frame+.2*jnp.tanh(corrected @ params["rgb_readout"]),0,1)
        return (corrected,predicted),None
    state,_ = jax.lax.scan(observe,(hidden,zero),jnp.swapaxes(context,0,1))
    def dream(state,_):
        hidden, frame = state
        updated = tick(params, hidden, jnp.concatenate((frame,zero),-1), "rgb")
        future = jnp.clip(frame+.2*jnp.tanh(updated @ params["rgb_readout"]),0,1)
        return (updated,future),future
    _,result=jax.lax.scan(dream,state,None,length=horizon)
    return jnp.swapaxes(result,0,1)


def signal_loss(params, pulse, dest, sign, ticks=10):
    prediction = signal(params,pulse,dest,ticks)
    return jnp.mean((prediction[:,-min(ticks,3):]-sign[:,None])**2)


def rgb_loss(params, episodes, context, horizon):
    predicted = rgb(params,episodes[:,:context],horizon)
    return jnp.mean((predicted-episodes[:,context:context+horizon])**2)
