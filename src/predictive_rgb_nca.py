"""RGB predictive neural cellular automaton (independent model family).

Shared local 3x3 cell rule, recurrent cell state, repeated cellular updates,
observation-error correction, and open-loop dream steps. No phase oscillators,
object labels, pose graph, or 3D reconstruction is assumed.
"""

import jax
import jax.numpy as jnp

from src.model import conv

CHANNELS = 12


def init(key, channels=CHANNELS):
    """Random cellular transition; identity RGB readout at initialization."""
    if channels < 1:
        raise ValueError("channels must be positive")
    a, b = jax.random.split(key)
    return {
        "perception": 0.05 * jax.random.normal(a, (3, 3, channels + 6, channels * 2)),
        "mix": 0.05 * jax.random.normal(b, (channels * 2, channels * 2)),
        "cell_bias": jnp.zeros((channels * 2,), dtype=jnp.float32),
        "rgb_readout": jnp.zeros((channels, 3), dtype=jnp.float32),
        "rgb_bias": jnp.zeros((3,), dtype=jnp.float32),
    }


def initial_state(params, batch, height, width):
    channels = params["rgb_readout"].shape[0]
    return (jnp.zeros((batch, height, width, channels), jnp.float32),
            jnp.zeros((batch, height, width, 3), jnp.float32))


def cell_update(params, hidden, image, surprise):
    """Same gated 3x3 rule at every pixel, with local messages and no special nodes."""
    perception = jnp.tanh(conv(jnp.concatenate((hidden, image, surprise), axis=-1),
                               params["perception"]))
    mixed = perception @ params["mix"] + params["cell_bias"]
    gate, proposal = jnp.split(mixed, 2, axis=-1)
    return hidden + 0.25 * jax.nn.sigmoid(gate) * jnp.tanh(proposal)


def observe(params, state, observed, ticks=2):
    """Correct internal state with *observed* RGB discrepancy, then clamp input.

    Surprise = observed image minus image forecast committed by previous dream.
    Current observation enters as input only, not as a hidden ground-truth target.
    """
    hidden, forecast = state
    surprise = observed - forecast

    def iterate(h, _):
        return cell_update(params, h, observed, surprise), None

    hidden, _ = jax.lax.scan(iterate, hidden, None, length=ticks)
    return hidden, observed


def dream(params, state, ticks=2):
    """Predict one frame without any new observation or correction signal."""
    hidden, image = state
    quiet = jnp.zeros_like(image)

    def iterate(h, _):
        return cell_update(params, h, image, quiet), None

    hidden, _ = jax.lax.scan(iterate, hidden, None, length=ticks)
    correction = 0.20 * jnp.tanh(hidden @ params["rgb_readout"] + params["rgb_bias"])
    future = jnp.clip(image + correction, 0.0, 1.0)
    return (hidden, future), future


def rollout_and_activity(params, observed_context, horizon, ticks=2):
    """[B,C,H,W,3] -> predicted RGB + mean absolute cellular activity.

    Only the context is seen; future target frames never enter dream(), at
    training or evaluation time. No observed frame arrives during dreaming.
    """
    if horizon < 1 or ticks < 1 or observed_context.ndim != 5 or observed_context.shape[-1] != 3:
        raise ValueError("Need RGB [B,C,H,W,3], positive horizon and ticks")
    batch, context, height, width, _ = observed_context.shape
    if context < 2:
        raise ValueError("Require at least two observed context frames")
    state = initial_state(params, batch, height, width)

    def absorb(s, frame):
        return observe(params, s, frame, ticks), None

    state, _ = jax.lax.scan(absorb, state, jnp.swapaxes(observed_context, 0, 1))

    def predict(s, _):
        s, frame = dream(params, s, ticks)
        activity = jnp.mean(jnp.abs(s[0]), axis=-1, keepdims=True)
        return s, (frame, activity)

    _, (rgb, activity) = jax.lax.scan(predict, state, None, length=horizon)
    return jnp.swapaxes(rgb, 0, 1), jnp.swapaxes(activity, 0, 1)


def rollout(params, observed_context, horizon, ticks=2):
    return rollout_and_activity(params, observed_context, horizon, ticks)[0]


def loss(params, examples, context=6, horizon=5, ticks=2):
    """Motion-emphasized training across *open-loop* future predictions."""
    predictions = rollout(params, examples[:, :context], horizon, ticks)
    target = examples[:, context:context + horizon]
    motion = jnp.mean(jnp.abs(target - examples[:, context - 1:context]),
                      axis=-1, keepdims=True)
    weights = 1.0 + 8.0 * motion
    return jnp.mean(weights * jnp.square(predictions - target))
