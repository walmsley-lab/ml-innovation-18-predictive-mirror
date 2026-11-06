"""Coarse recurrent prediction/error correction with full-resolution RGB transport.

Reuses the RGB NCA cell rule and the existing nonwrapping motion candidates.
Only the hidden sheet, perception convolution and routing logits are coarse;
RGB transport and rendering still do dense work at the original resolution.
This is an engineering hypothesis, not a biological or efficiency claim.
"""

import jax
import jax.numpy as jnp

from src import predictive_rgb_nca as nca
from src import rgb_motion_transport as transport


def init(key, channels=8):
    cell_key, router_key = jax.random.split(key)
    return {
        **nca.init(cell_key, channels),
        "router": .03 * jax.random.normal(
            router_key, (channels, len(transport.OFFSETS))),
        "router_bias": jnp.array(
            (2.,) + (-2.,) * (len(transport.OFFSETS) - 1), jnp.float32),
    }


def pool(image, factor=2):
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("State scale must be a positive integer")
    if image.shape[1] % factor or image.shape[2] % factor:
        raise ValueError("Image height and width must be divisible by state scale")
    b, h, w, c = image.shape
    return image.reshape(b, h // factor, factor, w // factor, factor, c).mean((2, 4))


def expand(field, image):
    return jax.image.resize(
        field, (*image.shape[:3], field.shape[-1]), method="linear")


def initial_state(params, frame, factor=2):
    low = pool(frame, factor)
    hidden = jnp.zeros((*low.shape[:3], params["rgb_readout"].shape[0]),
                       dtype=frame.dtype)
    return hidden, jnp.zeros_like(frame)


def observe(params, state, observed, ticks=2, factor=2, feedback=True):
    """Assimilate RGB against the forecast committed before it arrived."""
    hidden, committed = state
    low = pool(observed, factor)
    prior = pool(committed, factor) if feedback else low
    hidden, _ = nca.observe(params, (hidden, prior), low, ticks)
    return hidden, observed


def dream(params, state, ticks=2, factor=2):
    """Advance memory, transport existing RGB, then add learned appearance."""
    hidden, current = state
    (hidden, _), _ = nca.dream(params, (hidden, pool(current, factor)), ticks)
    logits = hidden @ params["router"] + params["router_bias"]
    weights = jax.nn.softmax(expand(logits, current), axis=-1)
    moved = jnp.sum(transport.candidates(current) * weights[..., None], axis=-2)
    residual = expand(.20 * jnp.tanh(
        hidden @ params["rgb_readout"] + params["rgb_bias"]), current)
    future = jnp.clip(moved + residual, 0., 1.)
    return (hidden, future), future


def assimilate(params, context, ticks=2, factor=2, feedback=True):
    if (context.ndim != 5 or context.shape[-1] != 3
            or context.shape[1] < 2 or min(context.shape) < 1 or ticks < 1):
        raise ValueError("Need nonempty RGB [B,T,H,W,3], T>=2 and positive ticks")
    state = initial_state(params, context[:, 0], factor)

    def absorb(state, frame):
        corrected = observe(params, state, frame, ticks, factor, feedback)
        committed, _ = dream(params, corrected, ticks, factor)
        return committed, None

    state, _ = jax.lax.scan(absorb, state, jnp.swapaxes(context[:, :-1], 0, 1))
    return observe(params, state, context[:, -1], ticks, factor, feedback)


def rollout(params, context, horizon=10, ticks=2, factor=2, feedback=True):
    """Context-only inference; lead 1 is immediately after the final observation."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    state = assimilate(params, context, ticks, factor, feedback)

    def predict(state, _):
        return dream(params, state, ticks, factor)

    _, frames = jax.lax.scan(predict, state, None, length=horizon)
    return jnp.swapaxes(frames, 0, 1)
