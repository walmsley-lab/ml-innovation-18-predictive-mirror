"""Optional online local gain plasticity over a frozen recurrent predictive field."""

import jax.numpy as jnp

from src import model


def initial(size=16, channels=8):
    return (jnp.zeros((1, size, size, channels), jnp.float32),
            jnp.zeros((1, size, size, 1), jnp.float32))


def neighbors(hidden):
    """Four-neighbor mean with zero-padded boundaries (no wraparound)."""
    p = jnp.pad(hidden, ((0, 0), (1, 1), (1, 1), (0, 0)))
    return (p[:, :-2, 1:-1] + p[:, 2:, 1:-1] +
            p[:, 1:-1, :-2] + p[:, 1:-1, 2:]) / 4


def step(params, state, frame, action, plasticity_rate=0.0):
    """Shared weights remain frozen; a spatial local gain adapts during sensing."""
    hidden, gain = state
    nearby = neighbors(hidden)
    next_hidden, prediction = model.step(params, hidden + gain * nearby, frame, action)
    correlation = jnp.mean(next_hidden * nearby, axis=-1, keepdims=True)
    next_gain = jnp.clip(0.99 * gain + plasticity_rate * correlation, -0.25, 0.25)
    return (next_hidden, next_gain), prediction
