"""Intervene on an NCA *after* observed context and before future prediction."""
import jax
import jax.numpy as jnp
from src import predictive_rgb_nca as nca
from src import rgb_grounded_nca as grounded

MODES=("intact","reset_hidden","shift_hidden","shock_image")


def observed_state(params,context,ticks=2,output='residual'):
    if context.ndim!=5 or context.shape[1]<2 or context.shape[-1]!=3 or ticks<1:
        raise ValueError("Need RGB context [B,T,H,W,3] and positive ticks")
    if output not in grounded.OUTPUT_MODES:
        raise ValueError('Unknown output head')
    batch,_,height,width,_=context.shape
    state=nca.initial_state(params,batch,height,width)
    def update(state,frame):
        corrected=nca.observe(params,state,frame,ticks)
        return grounded.dream(params,corrected,ticks,output)[0],None
    state,_=jax.lax.scan(update,state,jnp.swapaxes(context[:,:-1],0,1))
    return nca.observe(params,state,context[:,-1],ticks)


def perturb(state,mode):
    if mode not in MODES:
        raise ValueError("Unknown state manipulation")
    h,image=state
    if mode=="reset_hidden":
        return jnp.zeros_like(h),image
    if mode=="shift_hidden":
        return jnp.roll(h,2,axis=2),image
    if mode=="shock_image":
        return h,jnp.full_like(image,.5)
    return state


def forecast(params,context,horizon=10,ticks=2,mode="intact",output='residual'):
    if horizon<1:
        raise ValueError("Positive horizon required")
    state=perturb(observed_state(params,context,ticks,output),mode)
    def step(state,_):
        state,frame=grounded.dream(params,state,ticks,output)
        return state,frame
    _,frames=jax.lax.scan(step,state,None,length=horizon)
    return jnp.swapaxes(frames,0,1)
