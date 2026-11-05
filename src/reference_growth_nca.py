"""Minimal JAX NCA pattern-growth reference *exercise*.

Mordvintsev-style shared local recurrent updates; deliberately not a faithful
reproduction of the published paper's full training/damage curriculum.
"""
import jax
import jax.numpy as jnp
import numpy as np
from src.model import conv


def init(key, channels=8, width=24):
    if channels < 4 or width < 4:
        raise ValueError("Need >=4 state channels and update width")
    k1, k2 = jax.random.split(key)
    return {
        "perception": jax.random.normal(k1, (3,3,channels,width)) * .08,
        "update": jax.random.normal(k2, (width,channels)) * .02,
    }


def seed(batch=1, size=24, channels=8):
    if batch < 1 or size < 8 or channels < 4:
        raise ValueError("Invalid seed geometry")
    state = jnp.zeros((batch,size,size,channels), jnp.float32)
    return state.at[:,size//2,size//2,3].set(1.0)


def update(params, state, rate=.15):
    if rate <= 0:
        raise ValueError("Positive update rate required")
    perception = jax.nn.relu(conv(state,params["perception"]))
    delta = jnp.tanh(perception @ params["update"])
    return state + rate * delta


def rollout(params, initial, steps=16, rate=.15):
    if steps < 1:
        raise ValueError("Need at least one recurrent update")
    final,_ = jax.lax.scan(lambda state,_:(update(params,state,rate),None),
                           initial,None,length=steps)
    return final


def rgb(state):
    # Continuous color readout, not interpreted object identity.
    return jax.nn.sigmoid(state[...,:3]-2.0)


def target(size=24):
    if size < 8:
        raise ValueError("Size must be >=8")
    yy,xx = np.mgrid[:size,:size]
    rr = np.sqrt((xx-size//2)**2+(yy-size//2)**2)
    mask = (rr > size*.13)&(rr<size*.28)
    out = np.zeros((size,size,3),np.float32)
    out[mask] = np.array([.95,.35,.12],np.float32)
    return jnp.asarray(out)


def damage(state, fraction=.25):
    """Evaluator-only rectangle lesion; no claim of biological regeneration."""
    if not 0 <= fraction <= 1:
        raise ValueError("Invalid damage fraction")
    w = state.shape[2]
    pixels = round(w*fraction)
    return state.at[:,:,w//2:w//2+pixels,:].set(0)
