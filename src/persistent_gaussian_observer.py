"""RGB-only recurrent observer for compact Gaussian scene elements.

Unlike two-frame state inference, this observer propagates center/velocity
through ALL observed frames. Confidence from *observed pixels only* gates
corrections. Camera x/yaw commands are known and integrated causally.
"""
import jax
import jax.numpy as jnp
from src import rgb_gaussian_nca as gaussian
from src import gaussian_cellular_dynamics as dynamics


def visibility_confidence(frame,colors):
    """Approximate per-color observation reliability; gray occluders are ignored."""
    distance=jnp.sum((frame[...,None,:]-colors)**2,axis=-1)
    bright=(jnp.max(frame,axis=-1)>.25)[...,None]
    support=jnp.exp(-18.*distance)*bright
    return jnp.clip(jnp.sum(support,axis=(0,1))/12.,0.,1.)


def observe(params,history,actions,K,reset_at=-1):
    """Return (last_center, last_velocity, scale, colors, opacity, trajectory).

    History [context,H,W,RGB] must contain observed frames only; actions
    [context-1,2] are observed camera transitions, never future.
    """
    if (history.ndim!=4 or len(history)<2 or history.shape[-1]!=3
            or actions.shape!=(len(history)-1,2)):
        raise ValueError("Require observed RGB and aligned past camera actions")
    x,yaw=gaussian.camera_traces(actions)
    count=params["color_logits"].shape[0]
    zeros=jnp.zeros((count,3),jnp.float32)
    def scan(state,inputs):
        center,velocity=state
        t,frame,camera_x,camera_yaw=inputs
        observed,scale,colors,opacity=gaussian.infer(
            params,frame,K,camera_x,camera_yaw)
        confidence=visibility_confidence(frame,colors)
        trust=.65*confidence[:,None]
        predicted=center+velocity
        corrected=predicted+trust*(observed-predicted)
        speed=.75*velocity+.25*confidence[:,None]*(observed-center)
        first=(t==0)|(t==reset_at)
        result=jnp.where(first,observed,corrected)
        next_velocity=jnp.where(first,jnp.zeros_like(speed),speed)
        return (result,next_velocity),(
            result,next_velocity,confidence,scale,colors,opacity)
    end,(centers,velocities,conf,scales,colors,opacity)=jax.lax.scan(
        scan,(zeros,zeros),
        (jnp.arange(len(history)),history,x,yaw))
    return (*end,scales[-1],colors[-1],opacity[-1],
            {"centers":centers,"velocities":velocities,
             "confidence":conf})


def single(params,observations,actions,K,horizon=5,mode="cellular",
           reset_at=-1):
    if (observations.ndim!=4 or horizon<1
            or actions.shape!=(len(observations)+horizon-1,2)
            or mode not in gaussian.MODES):
        raise ValueError("Need context, planned future camera actions, mode")
    context=len(observations)
    center,velocity,scale,colors,opacity,_=observe(
        params,observations,actions[:context-1],K,reset_at)
    x,yaw=gaussian.camera_traces(actions)
    if mode=="static":
        forecast=jnp.broadcast_to(center,(horizon,*center.shape))
    elif mode=="velocity":
        forecast=center[None]+jnp.arange(
            1,horizon+1,dtype=jnp.float32)[:,None,None]*velocity[None]
    else:
        forecast=dynamics.rollout(params["dynamics"],center[None],
                                  velocity[None],horizon)[0]
    return jax.vmap(lambda centers,px,py:gaussian.render(
        centers,scale,colors,opacity,K,px,py,observations.shape[1]))(
            forecast,x[context:context+horizon],yaw[context:context+horizon])


def rollout(params,contexts,actions,K,horizon=5,mode="cellular",
            reset_at=-1):
    if (contexts.ndim!=5 or actions.ndim!=3
            or actions.shape!=(len(contexts),contexts.shape[1]+horizon-1,2)):
        raise ValueError("Expected [B,C,H,W,3] and [B,C+H-1,2]")
    return jax.vmap(lambda rgb,a:single(
        params,rgb,a,K,horizon,mode,reset_at))(contexts,actions)
