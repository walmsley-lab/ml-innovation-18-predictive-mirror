"""Fair ablations of fine/coarse bidirectional communication in fields.

Same convolutional core and readout in all modes. Coarse branch is
explicitly resampled (average downsample / nearest upsample), rather
than merely keeping another full-resolution hidden tensor.
"""
import jax
import jax.numpy as jnp
from src.model import conv

MODES=("fine","extra_ticks","dilated","encoder_decoder",
       "bidirectional","no_topdown","no_bottomup")


def init(key,channels=8):
    if channels<1:
        raise ValueError("Positive hidden width required")
    a,b,c,d,e=jax.random.split(key,5)
    return {"input":.06*jax.random.normal(a,(3,3,2,channels)),
            "core":.06*jax.random.normal(b,(3,3,channels,channels)),
            "up":.06*jax.random.normal(c,(channels,channels)),
            "down":.06*jax.random.normal(d,(channels,channels)),
            "readout":.03*jax.random.normal(e,(channels,2))}


def average2(x):
    b,h,w,c=x.shape
    if h%2 or w%2:
        raise ValueError("Multiscale fields require even H,W")
    return x.reshape(b,h//2,2,w//2,2,c).mean(axis=(2,4))


def up2(x):
    return jnp.repeat(jnp.repeat(x,2,axis=1),2,axis=2)


def dilation(x,kernel):
    import jax.lax as lax
    return lax.conv_general_dilated(
        x,kernel,window_strides=(1,1),padding="SAME",
        rhs_dilation=(2,2),dimension_numbers=("NHWC","HWIO","NHWC"))


def one(params,field,mode="bidirectional",ticks=2):
    if mode not in MODES or ticks<1 or field.ndim!=4 or field.shape[-1]!=2:
        raise ValueError("Invalid multiscale field inputs")
    fine=jnp.tanh(conv(field,params["input"]))
    coarse=average2(fine) if mode not in ("fine","extra_ticks","dilated") else None
    iters=ticks*2 if mode=="extra_ticks" else ticks
    for _ in range(iters):
        original=fine
        fine=jnp.tanh(conv(fine,params["core"]))
        if mode=="dilated":
            fine=jnp.tanh(fine+dilation(original,params["core"]))
        if coarse is not None:
            if mode=="bidirectional":
                coarse=jnp.tanh(conv(coarse,params["core"])+
                                average2(original)@params["down"])
                fine=fine+up2(coarse@params["up"])
            elif mode=="no_topdown":
                coarse=jnp.tanh(conv(coarse,params["core"])+
                                average2(original)@params["down"])
            elif mode=="no_bottomup":
                coarse=jnp.tanh(conv(coarse,params["core"]))
                fine=fine+up2(coarse@params["up"])
            else:
                coarse=jnp.tanh(conv(coarse,params["core"]))
    if mode=="encoder_decoder":
        fine=fine+up2(coarse@params["up"])
    return field+.08*jnp.tanh(fine@params["readout"])


def rollout(params,initial,horizon,mode="bidirectional",ticks=2):
    if horizon<1:
        raise ValueError("Invalid horizon")
    def advance(state,_):
        state=one(params,state,mode,ticks)
        return state,state
    _,result=jax.lax.scan(advance,initial,None,length=horizon)
    return jnp.swapaxes(result,0,1)


def loss(params,initial,truth,mode,ticks=2):
    output=rollout(params,initial,truth.shape[1],mode,ticks)
    return jnp.mean(jnp.square(output-truth))


def intervention(params,initial,horizon,ticks=2):
    """Ablations of each direction on an IDENTICAL frozen parameter set."""
    return {mode:rollout(params,initial,horizon,mode,ticks)
            for mode in ("bidirectional","no_topdown","no_bottomup",
                         "encoder_decoder","extra_ticks","dilated","fine")}
