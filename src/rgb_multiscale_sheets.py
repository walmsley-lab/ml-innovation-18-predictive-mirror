"""Independent fast/slow RGB cellular sheet experiment.

Both arms share the SAME parameter dictionary. The fast sheet updates every
tick; the optional slow sheet updates every second tick and feeds the fast
sheet through a trainable projection. No representation labels are supplied.
"""
import jax
import jax.numpy as jnp
from src import predictive_rgb_nca as base

MODES = ("single", "multiscale")


def init(key, channels=8):
    key, feedback = jax.random.split(key)
    params = base.init(key, channels)
    return {**params,
            "feedback": .05 * jax.random.normal(feedback, (channels, channels))}


def initial(params, batch, height, width):
    c = params["rgb_readout"].shape[0]
    zeros = jnp.zeros((batch, height, width, c), jnp.float32)
    image = jnp.zeros((batch, height, width, 3), jnp.float32)
    return zeros, zeros, image


def advance(params, state, image, error, ticks, mode, observe):
    fast, slow, _ = state

    def update(carry, tick):
        fast, slow = carry
        if mode == "multiscale":
            message = .08 * jnp.tanh(slow @ params["feedback"])
            new_fast = base.cell_update(params, fast + message, image, error)
            new_slow = base.cell_update(params, slow, image, error)
            slow = jnp.where((tick % 2) == 0, new_slow, slow)
        else:
            new_fast = base.cell_update(params, fast, image, error)
        return (new_fast, slow), None

    (fast, slow), _ = jax.lax.scan(
        update, (fast, slow), jnp.arange(ticks))
    combined = fast + (.35 * slow if mode == "multiscale" else 0.)
    correction = .2 * jnp.tanh(
        combined @ params["rgb_readout"] + params["rgb_bias"])
    forecast = jnp.clip(image + correction, 0, 1)
    # An observation clamps the image used for subsequent prediction;
    # dreaming commits the RGB forecast without future observations.
    return fast, slow, (image if observe else forecast)


def rollout(params, context, horizon=10, ticks=2, mode="multiscale"):
    if (mode not in MODES or context.ndim != 5 or context.shape[-1] != 3
            or context.shape[1] < 2 or horizon < 1 or ticks < 1):
        raise ValueError("Need RGB context, valid mode and positive horizon/ticks")
    batch, _, height, width, _ = context.shape
    state = initial(params, batch, height, width)

    def observe_and_predict(current, observed):
        previous = current[2]
        # First observe: update with sensory prediction error.
        current = advance(params, current, observed, observed-previous,
                          ticks, mode, True)
        # Commit next-frame prediction before subsequent sensory evidence.
        current = advance(params, current, observed, jnp.zeros_like(observed),
                          ticks, mode, False)
        return current, None

    state, _ = jax.lax.scan(
        observe_and_predict, state, jnp.swapaxes(context[:, :-1], 0, 1))
    last = context[:, -1]
    state = advance(params, state, last, last-state[2], ticks, mode, True)

    def dream(current, _):
        image = current[2]
        current = advance(params, current, image, jnp.zeros_like(image),
                          ticks, mode, False)
        return current, current[2]

    _, forecast = jax.lax.scan(dream, state, None, length=horizon)
    return jnp.swapaxes(forecast, 0, 1)


def loss(params, examples, context, horizon, ticks, mode):
    forecast = rollout(params, examples[:, :context], horizon, ticks, mode)
    target = examples[:, context:context+horizon]
    changed = jnp.mean(jnp.abs(target-examples[:, context-1:context]),
                       axis=-1, keepdims=True)
    return jnp.mean((1+8*changed)*jnp.square(forecast-target))
