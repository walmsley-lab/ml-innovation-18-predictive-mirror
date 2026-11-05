"""Tiny RGB-inferred Gaussian-state cellular forecaster.

A fixed number of learned color slots infer positions from *observed*
RGB moments. Camera pose is integrated from known planned x/yaw actions,
not read from simulator extrinsics. Dynamics are a shared 3D cellular rule.
No object labels, depth, future RGB, or future poses enter inference.
"""
import jax
import jax.numpy as jnp
from src import gaussian_cellular_dynamics as dynamics

MODES=("static","velocity","cellular")


def init(key,slots=3,width=8):
    if slots<1:
        raise ValueError("Need at least one slot")
    a,b=jax.random.split(key)
    rgb=jax.random.uniform(a,(slots,3),minval=.20,maxval=.80)
    return {
        "color_logits":jnp.log(rgb/(1-rgb)),
        "scale_logits":jnp.full((slots,),-1.1,jnp.float32),
        "opacity_logits":jnp.full((slots,),1.0,jnp.float32),
        "dynamics":dynamics.init(b,width),
    }


def rotation(yaw):
    c,s=jnp.cos(yaw),jnp.sin(yaw)
    return jnp.array([[c,0,s],[0,1,0],[-s,0,c]],jnp.float32)


def camera_traces(actions):
    """Camera pose at frame t: cumulative actions on transitions < t."""
    return (jnp.concatenate((jnp.zeros((1,),actions.dtype),
                             jnp.cumsum(actions[:,0]))),
            jnp.concatenate((jnp.zeros((1,),actions.dtype),
                             jnp.cumsum(actions[:,1]))))


def infer(params,frame,K,camera_x,camera_yaw):
    """Soft RGB slots -> metric 3D Gaussians (no privileged geometric inputs)."""
    size=frame.shape[0]
    yy,xx=jnp.mgrid[:size,:size]
    templates=jax.nn.sigmoid(params["color_logits"])
    foreground=jax.nn.sigmoid((jnp.max(frame,axis=-1)-.12)*55)
    proximity=-25*jnp.sum(
        (frame[...,None,:]-templates[None,None,:,:])**2,axis=-1)
    assignment=jax.nn.softmax(proximity,axis=-1)
    weights=assignment*foreground[...,None]
    norm=jnp.sum(weights,axis=(0,1))+1e-6
    u=jnp.sum(weights*xx[...,None],axis=(0,1))/norm
    v=jnp.sum(weights*yy[...,None],axis=(0,1))/norm
    spread2=jnp.sum(weights*((xx[...,None]-u)**2
                  +(yy[...,None]-v)**2),axis=(0,1))/(2*norm)
    sigma=jnp.sqrt(spread2+.2)
    scale=jax.nn.softplus(params["scale_logits"])+.05
    z=K[0,0]*scale/(sigma+.15)
    local=jnp.stack(((u-K[0,2])*z/K[0,0],
                     -(v-K[1,2])*z/K[1,1],z),axis=-1)
    world=local@rotation(camera_yaw).T+jnp.array([camera_x,0,0])
    return world,scale,templates,jax.nn.sigmoid(params["opacity_logits"])


def render(centers,scale,colors,opacity,K,camera_x,camera_yaw,size):
    """Differentiable soft Gaussian rasterizer, no hard RGB clip rails."""
    yy,xx=jnp.mgrid[:size,:size]
    local=(centers-jnp.array([camera_x,0,0]))@rotation(camera_yaw)
    z=jnp.maximum(local[:,2],.1)
    u=K[0,0]*local[:,0]/z+K[0,2]
    v=-K[1,1]*local[:,1]/z+K[1,2]
    sig=jnp.maximum(.3,K[0,0]*scale/z)
    distances=(xx[...,None]-u)**2+(yy[...,None]-v)**2
    contribution=opacity*jnp.exp(-.5*distances/sig**2)
    contribution=jnp.where(local[:,2]>0,contribution,0.)
    total=jnp.sum(contribution,axis=-1,keepdims=True)
    transmittance=jnp.exp(-total)
    mixture=jnp.sum(contribution[...,None]*colors,axis=-2)/(total+1e-6)
    return .035*transmittance+(1-transmittance)*mixture


def single(params,context_rgb,actions,K,horizon,mode):
    if (context_rgb.ndim!=4 or context_rgb.shape[-1]!=3
            or len(context_rgb)<2 or horizon<1
            or actions.shape!=(len(context_rgb)+horizon-1,2)
            or mode not in MODES):
        raise ValueError("Need RGB context and planned x/yaw actions")
    count=len(context_rgb)
    camera_x,camera_yaw=camera_traces(actions)
    prior,_,_,_=infer(params,context_rgb[-2],K,
                      camera_x[count-2],camera_yaw[count-2])
    current,scale,color,opacity=infer(
        params,context_rgb[-1],K,camera_x[count-1],camera_yaw[count-1])
    velocity=jnp.clip(current-prior,-.5,.5)
    if mode=="static":
        predicted=jnp.broadcast_to(current,(horizon,*current.shape))
    elif mode=="velocity":
        predicted=current[None]+jnp.arange(1,horizon+1)[:,None,None]*velocity
    else:
        predicted=dynamics.rollout(
            params["dynamics"],current[None],velocity[None],horizon)[0]
    views=jax.vmap(lambda p,x,y:render(
        p,scale,color,opacity,K,x,y,context_rgb.shape[1]))(
        predicted,camera_x[count:count+horizon],
        camera_yaw[count:count+horizon])
    # Context reconstruction uses ONLY the last observed frame.
    reconstruction=render(current,scale,color,opacity,K,
                          camera_x[count-1],camera_yaw[count-1],
                          context_rgb.shape[1])
    return views,reconstruction


def rollout(params,context_rgb,actions,K,horizon=5,mode="cellular"):
    """[B,C,H,W,3] + [B,C+H-1,2] + camera K -> [B,H,H,W,3]."""
    if (context_rgb.ndim!=5 or actions.ndim!=3
            or actions.shape!=(len(context_rgb),context_rgb.shape[1]+horizon-1,2)):
        raise ValueError("Invalid batched RGB / planned actions")
    forecast,_=jax.vmap(lambda rgb,a:single(
        params,rgb,a,K,horizon,mode))(context_rgb,actions)
    return forecast


def loss(params,episodes,actions,K,context=5,horizon=5):
    if episodes.shape[1]<context+horizon:
        raise ValueError("Observed sequence too short")
    forecasts,reconstruction=jax.vmap(
        lambda rgb,a:single(params,rgb[:context],a,K,horizon,"cellular"))(
            episodes,actions)
    truth=episodes[:,context:context+horizon]
    # Foreground weight uses *observed training target RGB*, not object IDs.
    foreground=jax.nn.sigmoid((jnp.max(truth,axis=-1,keepdims=True)-.12)*40)
    predictive=jnp.mean((1+4*foreground)*(forecasts-truth)**2)
    reconstruction_loss=jnp.mean(
        (reconstruction-episodes[:,context-1])**2)
    return predictive+.25*reconstruction_loss
