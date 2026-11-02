"""Causal interventions on the frozen RGB NCA's recurrent state.

Identical observed context and parameters; perturb hidden state *after*
observing context or after an already generated prediction. No retraining.
"""
import jax
import jax.numpy as jnp

from src import predictive_rgb_nca as nca

MODES = ("intact", "reset", "spatial_shuffle")


def perturb(state, mode):
    hidden, rgb = state
    if mode == "reset":
        hidden = jnp.zeros_like(hidden)
    elif mode == "spatial_shuffle":
        # Deterministic spatial displacement: preserve per-channel values
        # but break correspondence between hidden memory and image locations.
        hidden = jnp.roll(hidden, shift=(hidden.shape[1]//2,
                                         hidden.shape[2]//2), axis=(1, 2))
    elif mode != "intact":
        raise ValueError("Unknown hidden-state intervention")
    return hidden, rgb


def warm_state(params, observed_context, ticks=2):
    if (observed_context.ndim != 5 or observed_context.shape[1] < 2
            or observed_context.shape[-1] != 3 or ticks < 1):
        raise ValueError("Need RGB [B,context>=2,H,W,3] and ticks>=1")
    batch, _, height, width, _ = observed_context.shape
    state = nca.initial_state(params, batch, height, width)
    def step(current, frame):
        current = nca.observe(params, current, frame, ticks)
        current, _ = nca.dream(params, current, ticks)
        return current, None
    state, _ = jax.lax.scan(step, state,
                           jnp.swapaxes(observed_context[:, :-1], 0, 1))
    return nca.observe(params, state, observed_context[:, -1], ticks)


def forecast(params, observed_context, horizon=10, ticks=2,
             mode="intact", intervene_at=0):
    """When intervene_at=k, intervene before forecasting lead k+1.

    k=0 means just after observing context. For k>0, k original
    forecasts already happened; subsequent dreams have no observations.
    """
    if mode not in MODES or horizon < 1 or not 0 <= intervene_at < horizon:
        raise ValueError("Unknown mode or invalid intervention lead")
    state = warm_state(params, observed_context, ticks)
    if intervene_at == 0:
        state = perturb(state, mode)

    def dream(current, index):
        current = jax.lax.cond(
            (index == intervene_at) & (intervene_at != 0),
            lambda s: perturb(s, mode), lambda s: s, current)
        return nca.dream(params, current, ticks)

    _, future = jax.lax.scan(
        dream, state, jnp.arange(horizon))
    return jnp.swapaxes(future, 0, 1)
