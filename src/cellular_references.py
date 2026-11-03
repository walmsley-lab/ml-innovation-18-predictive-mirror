"""Tiny content-addressed cellular message routing; no provided object IDs."""
import jax
import jax.numpy as jnp

MODES=("uniform","learned","shuffled")


def init(key,channels=8):
    if channels<1:
        raise ValueError("channels must be positive")
    a,b,c,d,e,f=jax.random.split(key,6)
    features=channels+3
    return {
        "queries":.12*jax.random.normal(a,(features,channels)),
        "keys":.12*jax.random.normal(b,(features,channels)),
        "sender":.10*jax.random.normal(c,(features,channels)),
        "receiver":.10*jax.random.normal(d,(channels,channels)),
        "readout":.03*jax.random.normal(e,(channels,1)),
        "bias":jnp.zeros((channels,),jnp.float32),
    }


def update(params,hidden,pulse,target,mode="learned"):
    if mode not in MODES:
        raise ValueError("Unknown routing mode")
    batch,height,width,channels=hidden.shape
    fields=jnp.concatenate((hidden,pulse,jnp.abs(pulse),target),axis=-1)
    flattened=fields.reshape(batch,height*width,-1)
    query=flattened@params["queries"]
    keys=flattened@params["keys"]
    logits=4*jnp.einsum("bic,bjc->bij",query,keys)/jnp.sqrt(float(channels))
    weights=jax.nn.softmax(logits,axis=-1)
    if mode=="uniform":
        weights=jnp.ones_like(weights)/(height*width)
    elif mode=="shuffled":
        weights=jnp.roll(weights,7,axis=-1)
    message=flattened@params["sender"]
    selected=jnp.einsum("bij,bjc->bic",weights,message)
    selected=selected.reshape(batch,height,width,channels)
    updated=.7*hidden+.3*jnp.tanh(selected@params["receiver"]+params["bias"])
    return updated,weights


def rollout(params,pulse,target,ticks=4,mode="learned",return_routing=False):
    if (pulse.ndim!=4 or pulse.shape!=target.shape or pulse.shape[-1]!=1
            or ticks<1):
        raise ValueError("Need pulse and target [B,H,W,1]")
    batch,height,width,_=pulse.shape
    hidden=jnp.zeros((batch,height,width,params["readout"].shape[0]),jnp.float32)
    def advance(hidden,index):
        input_pulse=jnp.where(index==0,pulse,jnp.zeros_like(pulse))
        hidden,weights=update(params,hidden,input_pulse,target,mode)
        prediction=jnp.tanh(hidden@params["readout"])
        return hidden,(prediction,weights)
    _,(frames,weights)=jax.lax.scan(advance,hidden,jnp.arange(ticks))
    target_pred=jnp.sum(frames[-1]*target,axis=(1,2,3))
    if return_routing:
        return target_pred,weights[0]
    return target_pred


def loss(params,pulse,target,sign,ticks=4,mode="learned"):
    predicted=rollout(params,pulse,target,ticks,mode)
    return jnp.mean((predicted-sign)**2)
