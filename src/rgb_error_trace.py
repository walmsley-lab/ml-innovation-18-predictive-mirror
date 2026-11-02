"""Frozen RGB NCA plus local transient sensory-error trace.

This is within-episode *state adaptation*, not learned synaptic weights.
The trace is updated from observed prediction errors and decays while dreaming.
All three arms use exactly the same frozen base model parameters.
"""
import jax
import jax.numpy as jnp
from src import predictive_rgb_nca as nca

MODES = ("frozen", "local_trace", "shuffled_trace")


def update_trace(trace, surprise, decay=0.82):
    """A bounded, independent trace per cell and RGB channel."""
    return decay * trace + (1.0 - decay) * jnp.clip(surprise, -1.0, 1.0)


def adapt(prediction, trace, mode, strength):
    if mode == "frozen":
        return prediction
    if mode == "local_trace":
        effect = trace
    elif mode == "shuffled_trace":
        # Same values, spatially mismatched to the evidence that created them.
        effect = jnp.roll(trace, (trace.shape[1] // 2, trace.shape[2] // 2),
                          axis=(1, 2))
    else:
        raise ValueError("Unknown trace mode")
    return jnp.clip(prediction + strength * effect, 0.0, 1.0)


def rollout(params, context, horizon=10, ticks=2, mode="local_trace",
            decay=0.82, strength=0.12):
    """No future RGB input; state adaptation stops when observations stop."""
    if (mode not in MODES or context.ndim != 5 or context.shape[-1] != 3
            or context.shape[1] < 2 or horizon < 1 or ticks < 1
            or not 0 <= decay < 1 or strength < 0):
        raise ValueError("Invalid RGB context, trace mode, or settings")
    batch, _, height, width, _ = context.shape
    state = nca.initial_state(params, batch, height, width)
    trace = jnp.zeros((batch, height, width, 3), jnp.float32)

    def observed_then_predict(carry, observed):
        state, trace = carry
        # Surprise is predicted-before-observed, without a future label.
        trace = update_trace(trace, observed - state[1], decay)
        state = nca.observe(params, state, observed, ticks)
        state, predicted = nca.dream(params, state, ticks)
        state = (state[0], adapt(predicted, trace, mode, strength))
        return (state, trace * decay), None

    (state, trace), _ = jax.lax.scan(
        observed_then_predict, (state, trace),
        jnp.swapaxes(context[:, :-1], 0, 1))
    last = context[:, -1]
    trace = update_trace(trace, last - state[1], decay)
    state = nca.observe(params, state, last, ticks)

    def dream(carry, _):
        state, trace = carry
        state, prediction = nca.dream(params, state, ticks)
        future = adapt(prediction, trace, mode, strength)
        return ((state[0], future), trace * decay), future

    _, predictions = jax.lax.scan(
        dream, (state, trace), None, length=horizon)
    return jnp.swapaxes(predictions, 0, 1)
