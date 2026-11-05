"""Shared neural cellular rule on local 3D Gaussian scene elements.

The model receives previous scene centers and their observed velocities
here, so it is an *oracle-state* dynamics test, not visual perception.
"""
import jax
import jax.numpy as jnp


def init(key,width=12):
    if width<1:
        raise ValueError("Positive cell width required")
    a,b,c,d=jax.random.split(key,4)
    return {
        "input":.08*jax.random.normal(a,(6,width)),
        "self":.08*jax.random.normal(b,(width,width)),
        "neighbor":.08*jax.random.normal(c,(width,width)),
        "acceleration":.03*jax.random.normal(d,(width,3)),
    }


def step(params,center,velocity,hidden,radius=1.8):
    """One local, shared update; adjacent Gaussians exchange hidden messages."""
    if radius<=0 or center.shape[-1]!=3 or velocity.shape!=center.shape:
        raise ValueError("Expected metric xyz positions/velocities")
    count=center.shape[0]
    dist2=jnp.sum((center[:,None,:]-center[None,:,:])**2,axis=-1)
    weights=jnp.exp(-dist2/(2*radius**2))*(1-jnp.eye(count))
    incoming=(weights@hidden)/(weights.sum(axis=-1,keepdims=True)+1e-6)
    features=jnp.concatenate((center/6,velocity/.2),axis=-1)
    updated=jnp.tanh(features@params["input"]+hidden@params["self"]
                     +incoming@params["neighbor"])
    velocity=velocity+.012*jnp.tanh(updated@params["acceleration"])
    return center+velocity,velocity,updated


def rollout(params,centers,velocities,horizon=5,radius=1.8):
    """[batch,N,3] observed state -> [batch,horizon,N,3] open-loop.

    No future centers or future images are accepted by this function.
    """
    if (centers.ndim!=3 or centers.shape[-1]!=3
            or velocities.shape!=centers.shape or horizon<1):
        raise ValueError("Expected batched xyz scene state")
    width=params["self"].shape[0]
    def single(center,velocity):
        hidden=jnp.zeros((len(center),width),jnp.float32)
        def advance(state,_):
            next_state=step(params,*state,radius=radius)
            return next_state,next_state[0]
        _,result=jax.lax.scan(advance,(center,velocity,hidden),None,
                               length=horizon)
        return result
    return jax.vmap(single)(centers,velocities)


def loss(params,observed_center,observed_velocity,future_centers,radius=1.8):
    horizon=future_centers.shape[1]
    prediction=rollout(params,observed_center,observed_velocity,horizon,radius)
    return jnp.mean((prediction-future_centers)**2)
