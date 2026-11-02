"""Independent RGB NCA experiment: heterogeneous phase and sensory entrainment.

Phase is an explicit nontrainable dynamical field modulating RGB correction.
No future image or hidden simulator state is ever used to advance it.
"""
import jax
import jax.numpy as jnp

from src import predictive_rgb_nca as base, rgb_phase_nca as prior

MODES = ("plain", "free", "coupled", "entrained")


def natural_frequency(height, width):
    y, x = jnp.indices((height, width))
    return (.25 + .015*jnp.sin(x*.7) + .02*jnp.cos(y*.4))[None,...,None]


def evolve(phase, image, surprise, mode):
    """Local phase dynamics and optional entrainment to observed prediction error."""
    if mode not in MODES:
        raise ValueError("Unknown phase arm")
    height, width = phase.shape[1:3]
    omega = natural_frequency(height, width)
    coupling = 0.22*prior.alignment(phase) if mode in ("coupled", "entrained") else 0.
    # Surprise is nonzero only when a new observation is received.
    drive = (.65*jnp.tanh(jnp.mean(surprise, axis=-1, keepdims=True))
             if mode == "entrained" else 0.)
    return phase + omega + coupling + drive


def dream(params, state, phase, ticks, mode):
    hidden, image = state
    quiet = jnp.zeros_like(image)
    def tick(current, _):
        return base.cell_update(params, current, image, quiet), None
    hidden, _ = jax.lax.scan(tick, hidden, None, length=ticks)
    phase = evolve(phase, image, quiet, mode)
    residual = .2*jnp.tanh(hidden @ params["rgb_readout"] + params["rgb_bias"])
    next_frame = jnp.clip(image + residual*(1+.4*jnp.sin(phase)), 0, 1)
    return (hidden, next_frame), phase, next_frame


def rollout(params, context, horizon=10, ticks=2, mode="entrained",
            return_phase=False):
    if (mode not in MODES or context.ndim != 5 or context.shape[-1] != 3
            or context.shape[1] < 2 or horizon < 1 or ticks < 1):
        raise ValueError("Requires RGB context, positive horizon/ticks and known mode")
    if mode == "plain":
        output = base.rollout(params, context, horizon, ticks)
        if return_phase:
            return output, jnp.zeros((*context.shape[:1], *context.shape[2:4], 1))
        return output
    batch, _, height, width, _ = context.shape
    state = base.initial_state(params, batch, height, width)
    phase = prior.initial_phase(batch, height, width)

    def absorb_and_predict(current, frame):
        state, phase = current
        surprise = frame - state[1]
        state = base.observe(params, state, frame, ticks)
        phase = evolve(phase, frame, surprise, mode)
        state, phase, _ = dream(params, state, phase, ticks, mode)
        return (state, phase), None

    (state, phase), _ = jax.lax.scan(
        absorb_and_predict, (state, phase),
        jnp.swapaxes(context[:, :-1], 0, 1))
    last = context[:, -1]
    surprise = last - state[1]
    state = base.observe(params, state, last, ticks)
    phase = evolve(phase, last, surprise, mode)

    def forecast(current, _):
        state, phase = current
        state, phase, rgb = dream(params, state, phase, ticks, mode)
        return (state, phase), rgb
    (state, phase), future = jax.lax.scan(
        forecast, (state, phase), None, length=horizon)
    images = jnp.swapaxes(future, 0, 1)
    return (images, phase) if return_phase else images


def loss(params, examples, context, horizon, ticks, mode):
    forecast = rollout(params, examples[:, :context], horizon, ticks, mode)
    target = examples[:, context:context+horizon]
    motion = jnp.mean(
        jnp.abs(target - examples[:, context-1:context]), axis=-1,
        keepdims=True)
    return jnp.mean((1+8*motion)*jnp.square(forecast-target))


def phase_order(phase):
    """Kuramoto-style global coherence; informative, not a performance score."""
    vector = jnp.exp(1j*phase)
    return jnp.abs(jnp.mean(vector, axis=(1, 2, 3))).mean()
