"""Two-scale RGB predictive-coding pilot with optional latent supervision."""
import jax
import jax.numpy as jnp
from src.model import conv

MODES=("flat_rgb","two_scale_rgb","two_scale_latent","two_scale_precision")


def init(key,channels=8):
    if channels<1:
        raise ValueError("channels must be positive")
    a,b,c,d,e,f=jax.random.split(key,6)
    return {
        "fast":.05*jax.random.normal(a,(3,3,channels+6,channels)),
        "slow":.05*jax.random.normal(b,(3,3,channels+6,channels)),
        "fast_readout":.02*jax.random.normal(c,(channels,3)),
        "slow_readout":.02*jax.random.normal(d,(channels,3)),
        "precision":.03*jax.random.normal(e,(channels,1)),
        "encoder":.05*jax.random.normal(f,(3,3,3,channels)),
    }


def pool(x):
    return jax.lax.reduce_window(x,0.,jax.lax.add,
        (1,2,2,1),(1,2,2,1),"VALID")/4.


def upsample(x,shape):
    return jax.image.resize(x,shape,method="nearest")


def initial_state(params,frame):
    batch,height,width,_=frame.shape
    if height%2 or width%2:
        raise ValueError("Need even-height, even-width RGB images")
    channels=params["fast_readout"].shape[0]
    return (jnp.zeros((batch,height,width,channels),jnp.float32),
            jnp.zeros((batch,height//2,width//2,channels),jnp.float32),
            jnp.zeros_like(frame))


def update(params,state,frame,error,mode):
    hidden,slow,_=state
    if mode=="two_scale_precision":
        # A learned sensory-error gain; not a calibrated uncertainty estimate.
        error=error*(.25+1.5*jax.nn.sigmoid(hidden@params["precision"]))
    proposed=jnp.tanh(conv(jnp.concatenate((hidden,frame,error),axis=-1),
                              params["fast"]))
    hidden=.7*hidden+.3*proposed
    if mode!="flat_rgb":
        slow_frame=pool(frame)
        slow_error=pool(error)
        proposed_slow=jnp.tanh(conv(
            jnp.concatenate((slow,slow_frame,slow_error),axis=-1),
            params["slow"]))
        slow=.8*slow+.2*proposed_slow
    return hidden,slow


def forecast(params,hidden,slow,frame,mode):
    delta=hidden@params["fast_readout"]
    if mode!="flat_rgb":
        delta=delta+upsample(slow@params["slow_readout"],frame.shape)
    return jnp.clip(frame+.20*jnp.tanh(delta),0,1)


def rollout(params,context,horizon=10,mode="two_scale_rgb",return_latent=False):
    if (mode not in MODES or context.ndim!=5 or context.shape[-1]!=3
            or context.shape[1]<2 or horizon<1):
        raise ValueError("Need RGB context [B,C,H,W,3] and valid arm/horizon")
    state=initial_state(params,context[:,0])
    def observe(state,image):
        hidden,slow,prior=state
        hidden,slow=update(params,state,image,image-prior,mode)
        pred=forecast(params,hidden,slow,image,mode)
        return (hidden,slow,pred),None
    state,_=jax.lax.scan(observe,state,jnp.swapaxes(context,0,1))
    def dream(state,_):
        hidden,slow,current=state
        hidden,slow=update(params,state,current,jnp.zeros_like(current),mode)
        future=forecast(params,hidden,slow,current,mode)
        return (hidden,slow,future),(future,hidden)
    _,(images,latents)=jax.lax.scan(dream,state,None,length=horizon)
    images=jnp.swapaxes(images,0,1)
    latents=jnp.swapaxes(latents,0,1)
    return (images,latents) if return_latent else images


def loss(params,episodes,context=8,horizon=10,mode="two_scale_rgb"):
    predicted,hidden=rollout(params,episodes[:,:context],horizon,mode,True)
    truth=episodes[:,context:context+horizon]
    pixel=jnp.mean((predicted-truth)**2)
    if mode!="two_scale_latent":
        return pixel
    # A stop-gradient teacher embedding from the held-out future; labels
    # enter the *training loss*, never the context or open-loop dream.
    target=jax.lax.stop_gradient(jnp.tanh(
        conv(truth.reshape((-1,*truth.shape[2:])),params["encoder"])))
    target=target.reshape(hidden.shape)
    return pixel+.10*jnp.mean((hidden-target)**2)
