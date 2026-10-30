"""Two recurrent sheets with limited cross-talk, without assigned semantics."""

import jax
import jax.numpy as jnp

from src.model import conv


FAST_CHANNELS = 6
SLOW_CHANNELS = 2


def init(key):
    fast_key, slow_key, readout_key = jax.random.split(key, 3)
    # 3x3 kernels on both sheets; dilated slow kernel samples a wider field.
    return {
        "fast": jax.random.normal(fast_key, (3, 3, FAST_CHANNELS + SLOW_CHANNELS + 2, FAST_CHANNELS)) * 0.08,
        "slow": jax.random.normal(slow_key, (3, 3, FAST_CHANNELS + SLOW_CHANNELS + 2, SLOW_CHANNELS)) * 0.08,
        "readout": jax.random.normal(readout_key, (1, 1, FAST_CHANNELS + SLOW_CHANNELS, 1)) * 0.08,
        "bias": jnp.array(-2.5),
    }


def step(params, state, frame, action, feedback_strength=1.0):
    """One recurrent update. Feedback can be cut after training for ablations."""
    fast, slow = state
    pan = jnp.ones_like(frame) * action.reshape(1, 1, 1, 1)
    fast_in = jnp.concatenate((frame, pan, fast, feedback_strength * slow), axis=-1)
    slow_in = jnp.concatenate((frame, pan, slow, feedback_strength * fast), axis=-1)
    next_fast = 0.6 * fast + 0.4 * jnp.tanh(conv(fast_in, params["fast"]))
    next_slow = 0.85 * slow + 0.15 * jnp.tanh(jax.lax.conv_general_dilated(
        slow_in, params["slow"], (1, 1), "SAME", rhs_dilation=(2, 2),
        dimension_numbers=("NHWC", "HWIO", "NHWC")))
    combined = jnp.concatenate((next_fast, next_slow), axis=-1)
    forecast = jax.nn.sigmoid(conv(combined, params["readout"]) + params["bias"])
    return (next_fast, next_slow), forecast


def trace(params, frames, actions, feedback_strength=1.0):
    """Return next-frame predictions, fast states and slow states for probes."""
    h, w = frames.shape[1:3]
    start = (jnp.zeros((1, h, w, FAST_CHANNELS), jnp.float32),
             jnp.zeros((1, h, w, SLOW_CHANNELS), jnp.float32))

    def update(state, inputs):
        frame, action = inputs
        state, forecast = step(params, state, frame[None], action, feedback_strength)
        return state, (forecast[0], state[0][0], state[1][0])

    _, history = jax.lax.scan(update, start, (frames[:-1], actions))
    return history


def predict(params, frames, actions, feedback_strength=1.0):
    return trace(params, frames, actions, feedback_strength)[0]


def loss(params, frames, actions):
    forecast = jax.vmap(predict, in_axes=(None, 0, 0))(params, frames, actions)
    return jnp.mean((forecast - frames[:, 1:]) ** 2)
