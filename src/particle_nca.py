"""Small neural cellular automaton whose cells are moving 2D particles.

Inputs are PROVIDED 2D tracked particle positions, not extracted from RGB.
Pairs exchange learned hidden-state messages within a spatial radius.
Identical recurrent rule is shared across every particle.
"""
import jax
import jax.numpy as jnp


def init(key, channels=12):
    if channels < 1:
        raise ValueError("channels must be positive")
    a, b = jax.random.split(key)
    return {
        "update": .06*jax.random.normal(a, (4+2*channels, channels)),
        "bias": jnp.zeros((channels,), jnp.float32),
        "readout": jnp.zeros((channels, 2), jnp.float32),
    }


def initial_state(batch, particles, channels):
    return (jnp.zeros((batch, particles, channels), jnp.float32),
            jnp.zeros((batch, particles, 2), jnp.float32))


def neighbor_messages(hidden, positions, radius=.40):
    """Soft distance weights, zero self-connections; no supplied graph edges."""
    displacement = positions[:, :, None, :] - positions[:, None, :, :]
    sq = jnp.sum(displacement**2, axis=-1)
    count = positions.shape[1]
    weight = jnp.exp(-sq / (radius*radius))
    weight = weight * (1.0-jnp.eye(count, dtype=weight.dtype)[None])
    # Include an inert baseline in the denominator: with only two
    # particles, normalizing by sum(weight) alone would cancel distance.
    return jnp.einsum("bij,bjc->bic", weight, hidden) / (
        1.0 + jnp.sum(weight, axis=-1, keepdims=True))


def step(params, state, observed, mask, use_messages=True):
    """Observation mask=0 during dreaming: no true future positions enter."""
    hidden, previous_forecast = state
    anchor = mask*observed + (1-mask)*previous_forecast
    correction = mask*(observed-previous_forecast)
    messages = (neighbor_messages(hidden, anchor) if use_messages
                else jnp.zeros_like(hidden))
    features = jnp.concatenate((anchor, correction, hidden, messages), axis=-1)
    updated = .75*hidden + .25*jnp.tanh(
        features @ params["update"] + params["bias"])
    delta = .18*jnp.tanh(updated @ params["readout"])
    future = jnp.clip(anchor + delta, 0., 1.)
    return (updated, future), future


def rollout(params, observed_context, horizon=10, use_messages=True):
    """[B,context,N,2] -> [B,horizon,N,2]; open-loop future prediction."""
    if (observed_context.ndim != 4 or observed_context.shape[-1] != 2
            or observed_context.shape[1] < 2 or horizon < 1):
        raise ValueError("Need [B,context>=2,particles,2] and positive horizon")
    batch, _, particles, _ = observed_context.shape
    state = initial_state(batch, particles, params["readout"].shape[0])
    full = jnp.ones_like(observed_context[:, 0, :, :1])

    def absorb(current, frame):
        return step(params, current, frame, full, use_messages)
    state, _ = jax.lax.scan(
        absorb, state, jnp.swapaxes(observed_context, 0, 1))
    missing = jnp.zeros_like(observed_context[:, 0])
    no_mask = jnp.zeros_like(full)

    def dream(current, _):
        predicted = current[1]
        state, _ = step(params, current, missing, no_mask, use_messages)
        return state, predicted

    _, forecasts = jax.lax.scan(dream, state, None, length=horizon)
    return jnp.swapaxes(forecasts, 0, 1)


def loss(params, examples, context=8, horizon=10, use_messages=True):
    forecast = rollout(params, examples[:, :context], horizon, use_messages)
    targets = examples[:, context:context+horizon]
    return jnp.mean(jnp.square(forecast-targets))
