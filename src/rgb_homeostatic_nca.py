"""Independent local activity-regulation ablation for the RGB predictive NCA.

A bounded per-cell recovery trace supplies negative feedback. No fixed
harmonic forcing, oscillator, added trainable weights or future observations.
"""
import jax
import jax.numpy as jnp

from src import predictive_rgb_nca as nca

MODES = ("plain", "local_recovery", "shuffled_recovery")


def initial_state(params, batch, height, width):
    hidden, image = nca.initial_state(params, batch, height, width)
    return hidden, image, jnp.zeros_like(hidden)


def cell_tick(params, hidden, recovery, image, surprise, mode, strength, decay):
    next_hidden = nca.cell_update(params, hidden, image, surprise)
    # A bounded low-pass trace of local activity: 0 <= recovery <= 1.
    next_recovery = decay * recovery + (1 - decay) * jnp.square(
        jnp.tanh(next_hidden))
    if mode == "local_recovery":
        feedback = next_recovery
    elif mode == "shuffled_recovery":
        feedback = jnp.roll(next_recovery,
                            (hidden.shape[1] // 2, hidden.shape[2] // 2),
                            axis=(1, 2))
    else:
        feedback = jnp.zeros_like(next_recovery)
    regulated = next_hidden - strength * feedback * jnp.tanh(next_hidden)
    return regulated, next_recovery


def observe(params, state, observed, ticks, mode, strength, decay):
    hidden, forecast, recovery = state
    surprise = observed - forecast

    def tick(carry, _):
        h, r = carry
        return cell_tick(params, h, r, observed, surprise,
                         mode, strength, decay), None

    (hidden, recovery), _ = jax.lax.scan(
        tick, (hidden, recovery), None, length=ticks)
    return hidden, observed, recovery


def dream(params, state, ticks, mode, strength, decay):
    hidden, image, recovery = state
    quiet = jnp.zeros_like(image)

    def tick(carry, _):
        h, r = carry
        return cell_tick(params, h, r, image, quiet,
                         mode, strength, decay), None

    (hidden, recovery), _ = jax.lax.scan(
        tick, (hidden, recovery), None, length=ticks)
    correction = .20 * jnp.tanh(hidden @ params["rgb_readout"]
                                 + params["rgb_bias"])
    future = jnp.clip(image + correction, 0, 1)
    return (hidden, future, recovery), future


def rollout_and_diagnostics(params, context, horizon=10, ticks=2,
                            mode="local_recovery", strength=.15, decay=.9):
    if (mode not in MODES or context.ndim != 5 or context.shape[1] < 2
            or context.shape[-1] != 3 or horizon < 1 or ticks < 1
            or not 0 <= strength <= 1 or not 0 <= decay < 1):
        raise ValueError("Invalid RGB context, mode or regulation settings")
    batch, _, height, width, _ = context.shape
    state = initial_state(params, batch, height, width)

    def absorb(s, frame):
        state = observe(params, s, frame, ticks, mode, strength, decay)
        state, _ = dream(params, state, ticks, mode, strength, decay)
        return state, None

    state, _ = jax.lax.scan(
        absorb, state, jnp.swapaxes(context[:, :-1], 0, 1))
    state = observe(params, state, context[:, -1], ticks, mode, strength, decay)

    def predict(s, _):
        s, frame = dream(params, s, ticks, mode, strength, decay)
        hidden, _, recovery = s
        return s, (frame, jnp.mean(jnp.abs(hidden)),
                   jnp.max(jnp.abs(hidden)), jnp.max(recovery))

    _, (predictions, mean_abs, max_abs, max_recovery) = jax.lax.scan(
        predict, state, None, length=horizon)
    return (jnp.swapaxes(predictions, 0, 1), {
        "mean_abs_hidden": jnp.mean(mean_abs),
        "max_abs_hidden": jnp.max(max_abs),
        "max_recovery": jnp.max(max_recovery),
    })


def rollout(params, context, horizon=10, ticks=2,
            mode="local_recovery", strength=.15, decay=.9):
    return rollout_and_diagnostics(
        params, context, horizon, ticks, mode, strength, decay)[0]


def loss(params, examples, context=8, horizon=10, ticks=2,
         mode="local_recovery", strength=.15, decay=.9):
    predicted = rollout(params, examples[:, :context], horizon, ticks,
                        mode, strength, decay)
    truth = examples[:, context:context + horizon]
    motion = jnp.mean(
        jnp.abs(truth - examples[:, context - 1:context]), axis=-1,
        keepdims=True)
    return jnp.mean((1 + 8 * motion) * jnp.square(predicted - truth))
