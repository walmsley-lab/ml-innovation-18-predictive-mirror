"""Optional phase-coupled RGB cellular dynamics, with no added trainable weights.

Modes: baseline NCA, uncoupled spatial oscillators, locally coupled oscillators.
This is an independent hypothesis; oscillations do not imply better prediction.
"""
import jax
import jax.numpy as jnp
from src import predictive_rgb_nca as nca
from src.model import conv

MODES=("plain","uncoupled","coupled")


def initial_phase(batch, height, width):
    y,x=jnp.indices((height,width))
    spatial=.11*x+.07*y
    return jnp.broadcast_to(spatial[None,...,None],
                            (batch,height,width,1))


def alignment(phase):
    kernel=jnp.ones((3,3,1,1),phase.dtype)/8
    kernel=kernel.at[1,1,0,0].set(0.)
    sine,cosine=jnp.sin(phase),jnp.cos(phase)
    return cosine*conv(sine,kernel)-sine*conv(cosine,kernel)


def phase_step(phase, image, coupling=0):
    if coupling<0:
        raise ValueError("Coupling must be nonnegative")
    # Heterogeneous spatial starting phases prevent trivial zero alignment.
    luminance=jnp.mean(image,axis=-1,keepdims=True)
    return phase+.35+.12*(luminance-.5)+coupling*alignment(phase)


def dream(params, state, phase, ticks=2, coupling=0):
    hidden,image=state
    quiet=jnp.zeros_like(image)
    def tick(h,_):
        return nca.cell_update(params,h,image,quiet),None
    hidden,_=jax.lax.scan(tick,hidden,None,length=ticks)
    phase=phase_step(phase,image,coupling)
    # Preserve strict persistence at zero-initialized readout.
    correction=.20*jnp.tanh(hidden@params["rgb_readout"]+params["rgb_bias"])
    predicted=jnp.clip(image+correction*(1+.4*jnp.sin(phase)),0,1)
    return (hidden,predicted),phase,predicted


def rollout(params, observed, horizon=10, ticks=2, mode="plain"):
    if mode not in MODES or horizon<1 or ticks<1:
        raise ValueError("Unknown oscillator mode or invalid horizon/ticks")
    if mode=="plain":
        return nca.rollout(params,observed,horizon,ticks)
    if observed.ndim!=5 or observed.shape[-1]!=3 or observed.shape[1]<2:
        raise ValueError("Requires observed [B,C,H,W,3] RGB frames")
    batch,_,height,width,_=observed.shape
    state=nca.initial_state(params,batch,height,width)
    phase=initial_phase(batch,height,width)
    coupling=0.0 if mode=="uncoupled" else .25

    def absorb(s,frame):
        state,phase=s
        corrected=nca.observe(params,state,frame,ticks)
        phase=phase_step(phase,frame,coupling)
        predicted,phase,_=dream(params,corrected,phase,ticks,coupling)
        return (predicted,phase),None

    (state,phase),_=jax.lax.scan(
        absorb,(state,phase),jnp.swapaxes(observed[:,:-1],0,1))
    state=nca.observe(params,state,observed[:,-1],ticks)
    phase=phase_step(phase,observed[:,-1],coupling)

    def forecast(s,_):
        state,phase=s
        state,phase,rgb=dream(params,state,phase,ticks,coupling)
        return (state,phase),rgb

    _,images=jax.lax.scan(forecast,(state,phase),None,length=horizon)
    return jnp.swapaxes(images,0,1)


def loss(params,examples,context,horizon,ticks,mode):
    forecast=rollout(params,examples[:,:context],horizon,ticks,mode)
    truth=examples[:,context:context+horizon]
    motion=jnp.mean(jnp.abs(truth-examples[:,context-1:context]),
                    axis=-1,keepdims=True)
    return jnp.mean((1+8*motion)*jnp.square(forecast-truth))
