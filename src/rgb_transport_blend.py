"""Bounded transport-plus-synthesis comparison, reusing existing motion router.

Keep src.rgb_motion_transport unchanged as the reference. This variant
separates relocating observed RGB from synthesizing genuinely new appearance.
"""
import jax
import jax.numpy as jnp
from src import rgb_motion_transport as transport
from src.model import conv

MODES=("transport","bounded_hybrid")


def init(key,width=12):
    if width<1:
        raise ValueError("width must be positive")
    base=transport.init(key,width)
    extra=jax.random.normal(jax.random.fold_in(key,17),(width,1))*.02
    return {**base,"synthesis_gate":extra,
            "synthesis_bias":jnp.full((1,),-2.5,jnp.float32)}


def one_step(params,previous,current,mode="bounded_hybrid",return_parts=False):
    if mode not in MODES:
        raise ValueError("Unknown transport hybrid mode")
    features=jnp.concatenate((current,current-previous),axis=-1)
    hidden=jax.nn.relu(conv(features,params["perception"]))
    weights=jax.nn.softmax(hidden@params["router"]+params["router_bias"],axis=-1)
    moved=jnp.sum(transport.candidates(current)*weights[...,None],axis=-2)
    color=jax.nn.sigmoid(hidden@params["residual"]+params["residual_bias"])
    gate=jax.nn.sigmoid(hidden@params["synthesis_gate"]+params["synthesis_bias"])
    future=moved if mode=="transport" else (1-gate)*moved+gate*color
    return (future,{"transport":moved,"synthesized":color,
                    "synthesis_gate":gate}) if return_parts else future


def rollout(params,context,horizon=10,mode="bounded_hybrid"):
    if context.ndim!=5 or context.shape[-1]!=3 or context.shape[1]<2 or horizon<1:
        raise ValueError("Expected RGB [B,T,H,W,3] and positive horizon")
    def update(state,_):
        previous,current=state
        future=one_step(params,previous,current,mode)
        return (current,future),future
    _,forecast=jax.lax.scan(update,(context[:,-2],context[:,-1]),
                            None,length=horizon)
    return jnp.swapaxes(forecast,0,1)


def loss(params,samples,context=8,horizon=10,mode="bounded_hybrid"):
    predictions=rollout(params,samples[:,:context],horizon,mode)
    truth=samples[:,context:context+horizon]
    return jnp.mean((predictions-truth)**2)
