"""Learn local RGB transport and an optional cellular-style residual correction."""
import jax
import jax.numpy as jnp
from src.model import conv

MODES=("residual","transport","combined")
OFFSETS=((0,0),(-2,0),(2,0),(0,-2),(0,2),
         (-2,-2),(-2,2),(2,-2),(2,2))


def init(key,width=12):
    if width<1:
        raise ValueError("width must be positive")
    a,b,c=jax.random.split(key,3)
    prior=jnp.array((2.0,)+(-2.0,)*(len(OFFSETS)-1),jnp.float32)
    return {
        "perception":.06*jax.random.normal(a,(3,3,6,width)),
        "router":.03*jax.random.normal(b,(width,len(OFFSETS))),
        "router_bias":prior,
        "residual":.02*jax.random.normal(c,(width,3)),
        "residual_bias":jnp.zeros((3,),jnp.float32),
    }


def candidates(image):
    """Nonwrapping local warps. Each offset is a *destination* displacement."""
    batch,height,width,channels=image.shape
    pad=2
    padded=jnp.pad(image,((0,0),(pad,pad),(pad,pad),(0,0)))
    return jnp.stack([
        padded[:,pad-dy:pad-dy+height,pad-dx:pad-dx+width,:]
        for dy,dx in OFFSETS],axis=-2)


def one_step(params,previous,current,mode="combined",return_weights=False):
    if mode not in MODES:
        raise ValueError("Unknown forecaster arm")
    features=jnp.concatenate((current,current-previous),axis=-1)
    hidden=jax.nn.relu(conv(features,params["perception"]))
    weights=jax.nn.softmax(hidden@params["router"]+params["router_bias"],axis=-1)
    transported=jnp.sum(candidates(current)*weights[...,None],axis=-2)
    correction=.20*jnp.tanh(hidden@params["residual"]+params["residual_bias"])
    if mode=="residual":
        next_frame=jnp.clip(current+correction,0,1)
    elif mode=="transport":
        next_frame=jnp.clip(transported,0,1)
    else:
        next_frame=jnp.clip(transported+correction,0,1)
    return (next_frame,weights) if return_weights else next_frame


def rollout(params,context,horizon=10,mode="combined"):
    if (context.ndim!=5 or context.shape[-1]!=3 or
            context.shape[1]<2 or horizon<1):
        raise ValueError("Expected [B,C,H,W,3] and positive horizon")
    def predict(state,_):
        before,current=state
        future=one_step(params,before,current,mode)
        return (current,future),future
    _,result=jax.lax.scan(predict,(context[:,-2],context[:,-1]),None,length=horizon)
    return jnp.swapaxes(result,0,1)


def loss(params,samples,context=8,horizon=10,mode="combined"):
    predicts=rollout(params,samples[:,:context],horizon,mode)
    return jnp.mean((predicts-samples[:,context:context+horizon])**2)
