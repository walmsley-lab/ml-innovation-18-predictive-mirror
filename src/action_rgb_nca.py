"""Action-conditioned open-loop RGB NCA for calibrated perspective worlds.

The observed frames are the only visual inputs. Future actions are supplied
only when explicitly planned; future RGB/depth/pose never enter inference.
"""
import jax
import jax.numpy as jnp
from src.model import conv


def init(key,channels=8):
    if channels<1:
        raise ValueError("Invalid cellular width")
    a,b=jax.random.split(key)
    return {
        "perception":.06*jax.random.normal(a,(3,3,channels+8,2*channels)),
        "mix":.06*jax.random.normal(b,(2*channels,2*channels)),
        "bias":jnp.zeros((2*channels,),jnp.float32),
        "readout":.02*jax.random.normal(b,(channels,3)),
    }


def update(params,hidden,image,error,action):
    batch,height,width,_=image.shape
    if action.shape!=(batch,2):
        raise ValueError("Camera action must be [B,2]")
    tiled=jnp.broadcast_to(action[:,None,None,:],(batch,height,width,2))
    x=jnp.concatenate((hidden,image,error,tiled),axis=-1)
    observed=jnp.tanh(conv(x,params["perception"]))
    gates,proposal=jnp.split(observed@params["mix"]+params["bias"],2,axis=-1)
    return hidden+.2*jax.nn.sigmoid(gates)*jnp.tanh(proposal)


def observe(params,state,frame):
    hidden,prior=state
    zeros=jnp.zeros((frame.shape[0],2),frame.dtype)
    return update(params,hidden,frame,frame-prior,zeros),frame


def dream(params,state,action):
    hidden,frame=state
    hidden=update(params,hidden,frame,jnp.zeros_like(frame),action)
    future=jnp.clip(frame+.25*jnp.tanh(hidden@params["readout"]),0,1)
    return (hidden,future),future


def rollout(params,context,actions,horizon=5,aware=True):
    if (context.ndim!=5 or context.shape[-1]!=3 or context.shape[1]<2
            or horizon<1 or actions.shape!=(context.shape[0],
                      context.shape[1]+horizon-1,2)):
        raise ValueError("Need context [B,C,H,W,3] and planned [B,C+H-1,2] actions")
    if not aware:
        actions=jnp.zeros_like(actions)
    batch,count,height,width,_=context.shape
    hidden=jnp.zeros((batch,height,width,params["readout"].shape[0]),jnp.float32)
    state=observe(params,(hidden,jnp.zeros_like(context[:,0])),context[:,0])
    def absorb(state,pair):
        frame,action=pair
        state,_=dream(params,state,action)
        return observe(params,state,frame),None
    state,_=jax.lax.scan(absorb,state,(
        jnp.swapaxes(context[:,1:],0,1),
        jnp.swapaxes(actions[:,:count-1],0,1)))
    def predict(state,action):
        return dream(params,state,action)
    _,future=jax.lax.scan(
        predict,state,jnp.swapaxes(actions[:,count-1:],0,1))
    return jnp.swapaxes(future,0,1)


def loss(params,episode_frames,actions,context,horizon,aware=True):
    output=rollout(params,episode_frames[:,:context],actions,horizon,aware)
    return jnp.mean((output-episode_frames[:,context:context+horizon])**2)
