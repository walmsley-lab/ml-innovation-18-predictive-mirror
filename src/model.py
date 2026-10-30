"""One locally connected recurrent predictive sheet; functional JAX only."""

import jax
import jax.numpy as jnp


def conv(x, kernel):
    return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME",
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))


def init(key, channels=8):
    a, b = jax.random.split(key)
    return {
        "local": jax.random.normal(a, (3, 3, channels + 2, channels)) * 0.08,
        "readout": jax.random.normal(b, (1, 1, channels, 1)) * 0.08,
        "bias": jnp.array(-2.5),
    }


def step(params, hidden, frame, action):
    """Update cellular state from observed image and scalar camera action."""
    pan = jnp.ones_like(frame) * action.reshape(1, 1, 1, 1)
    x = jnp.concatenate([frame, pan, hidden], axis=-1)
    next_hidden = 0.6 * hidden + 0.4 * jnp.tanh(conv(x, params["local"]))
    forecast = jax.nn.sigmoid(conv(next_hidden, params["readout"]) + params["bias"])
    return next_hidden, forecast


def predict(params, frames, actions):
    """Teacher-forced predictions: observations 0..T-1 -> forecasts 1..T."""
    h, w = frames.shape[1:3]
    channels = params["local"].shape[-1]
    start = jnp.zeros((1, h, w, channels), jnp.float32)

    def move(state, inputs):
        frame, action = inputs
        next_state, out = step(params, state, frame[None], action)
        return next_state, out[0]

    _, outs = jax.lax.scan(move, start, (frames[:-1], actions))
    return outs


def loss(params, frames, actions, loss_mode="plain"):
    forecast = jax.vmap(predict, in_axes=(None, 0, 0))(params, frames, actions)
    target = frames[:, 1:]
    # A plain observation-only loss; simulator foreground masks are evaluation-only.
    return (jnp.mean((1.0 + 32.0 * (target > 0.9)) * (forecast - target) ** 2)
            if loss_mode == "contrast" else jnp.mean((forecast - target) ** 2))
