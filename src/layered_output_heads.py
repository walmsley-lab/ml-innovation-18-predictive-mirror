"""Matched observation budget for direct RGB, transport and synthesis.

All modes consume the same last three RGB frames. The common CNN predicts
motion and a residual readout. Flow and visibility oracle inputs are
available ONLY to the explicit evaluator probe, never to rollout/loss.
"""
import jax
import jax.numpy as jnp
from src.model import conv

MODES=("direct","transport","transport_synthesis")


def init(key,width=8):
    if width<1:raise ValueError("Positive width required")
    a,b,c,d,e=jax.random.split(key,5)
    return {"encode":.07*jax.random.normal(a,(3,3,9,width)),
            "core":.06*jax.random.normal(b,(3,3,width,width)),
            "flow":.04*jax.random.normal(c,(width,2)),
            "confidence":.04*jax.random.normal(d,(width,1)),
            "rgb":.04*jax.random.normal(e,(width,3))}


def gather(image,dyx):
    """Nonwrapping bilinear backward sampling in (dy,dx) convention."""
    image=jnp.asarray(image);dyx=jnp.asarray(dyx)
    if image.ndim!=4 or dyx.shape!=(*image.shape[:3],2):
        raise ValueError("Need [B,H,W,C] and [B,H,W,2]")
    b,h,w,c=image.shape
    yy,xx=jnp.mgrid[:h,:w]
    sy=yy[None]-dyx[...,0];sx=xx[None]-dyx[...,1]
    valid=(sy>=0)&(sy<=h-1)&(sx>=0)&(sx<=w-1)
    fy=jnp.floor(sy);fx=jnp.floor(sx)
    y0=jnp.clip(fy.astype(int),0,h-1);x0=jnp.clip(fx.astype(int),0,w-1)
    y1=jnp.minimum(y0+1,h-1);x1=jnp.minimum(x0+1,w-1)
    yfrac=jnp.clip(sy-fy,0,1)[...,None]
    xfrac=jnp.clip(sx-fx,0,1)[...,None]
    batch=jnp.arange(b)[:,None,None]
    sample=((1-yfrac)*(1-xfrac)*image[batch,y0,x0]+
            yfrac*(1-xfrac)*image[batch,y1,x0]+
            (1-yfrac)*xfrac*image[batch,y0,x1]+
            yfrac*xfrac*image[batch,y1,x1])
    return jnp.where(valid[...,None],sample,.035),valid


def predict(params,older,previous,current,mode="transport_synthesis",
            oracle_flow=None,oracle_visibility=None,return_aux=False):
    if mode not in MODES:
        raise ValueError("Invalid output head")
    if current.shape!=previous.shape or current.shape!=older.shape:
        raise ValueError("All observed RGB views must match")
    feature=jnp.concatenate((older,previous,current),axis=-1)
    hidden=jnp.tanh(conv(feature,params["encode"]))
    hidden=jnp.tanh(conv(hidden,params["core"]))
    estimated_flow=3*jnp.tanh(hidden@params["flow"])
    flow=estimated_flow if oracle_flow is None else oracle_flow
    # For t+1, t-1 content is displaced approximately twice as far.
    recent,mask_recent=gather(current,flow)
    old,mask_old=gather(previous,2*flow)
    confidence=jax.nn.sigmoid(hidden@params["confidence"])
    mixture=confidence*recent+(1-confidence)*old
    usable=jnp.logical_or(mask_recent,mask_old)
    blended=jnp.where(usable[...,None],mixture,.035)
    residual=.20*jnp.tanh(hidden@params["rgb"])
    if mode=="direct":
        next_rgb=jnp.clip(current+residual,0,1)
    elif mode=="transport":
        next_rgb=jnp.clip(blended,0,1)
    else:
        next_rgb=jnp.clip(blended+residual,0,1)
    if oracle_visibility is not None:
        # Evaluator-only forced visibility: suppress transport where
        # source pixels are unavailable; synthesis may still fill.
        allowed=jnp.asarray(oracle_visibility)[...,None]
        transported=jnp.where(allowed,blended,.035)
        next_rgb=(transported if mode=="transport" else
                  transported+residual if mode=="transport_synthesis" else next_rgb)
        next_rgb=jnp.clip(next_rgb,0,1)
    if return_aux:
        return next_rgb,{"predicted_flow":estimated_flow,
                         "flow_used":flow,"valid_from_history":usable,
                         "history_blend":confidence,"transport_rgb":blended}
    return next_rgb


def rollout(params,history,horizon=3,mode="transport_synthesis"):
    if history.ndim!=5 or history.shape[1]<3 or horizon<1:
        raise ValueError("Need RGB [B,T>=3,H,W,3] and positive horizon")
    older,previous,current=(history[:,-3],history[:,-2],history[:,-1])
    def advance(carry,_):
        old,prev,now=carry
        nxt=predict(params,old,prev,now,mode)
        return (prev,now,nxt),nxt
    _,out=jax.lax.scan(advance,(older,previous,current),None,length=horizon)
    return jnp.swapaxes(out,0,1)


def loss(params,examples,context,horizon,mode):
    forecast=rollout(params,examples[:,:context],horizon,mode)
    target=examples[:,context:context+horizon]
    motion=jnp.mean(abs(target-examples[:,context-1:context]),axis=-1,keepdims=True)
    return jnp.mean((1+4*motion)*(forecast-target)**2)
