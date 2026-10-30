"""Tiny recurrent next-frame residual model: persistence is the exact initialization."""

import jax
import jax.numpy as jnp
from src.model import conv

CHANNELS = 16


def init(key, channels=CHANNELS):
    return {
        "local": 0.04 * jax.random.normal(key, (3, 3, channels + 2, channels)),
        "readout": jnp.zeros((1, 1, channels, 1)),
        "bias": jnp.array(0.0),
    }


def step(params, hidden, frame, action):
    action_image = jnp.ones_like(frame) * action.reshape(1, 1, 1, 1)
    state = jnp.concatenate((frame, action_image, hidden), axis=-1)
    updated = 0.7 * hidden + 0.3 * jnp.tanh(conv(state, params["local"]))
    delta = 0.25 * jnp.tanh(conv(updated, params["readout"]) + params["bias"])
    return updated, jnp.clip(frame + delta, 0.0, 1.0)


def predict(params, frames, actions):
    height, width = frames.shape[1:3]
    start = jnp.zeros((1, height, width, params["local"].shape[-1]), jnp.float32)

    def loop(hidden, pair):
        frame, action = pair
        updated, next_frame = step(params, hidden, frame[None], action)
        return updated, next_frame[0]

    _, forecast = jax.lax.scan(loop, start, (frames[:-1], actions))
    return forecast


def loss(params, frames, actions):
    predicted = jax.vmap(predict, in_axes=(None, 0, 0))(params, frames, actions)
    target = frames[:, 1:]
    # Motion comes from adjacent observed images, never hidden simulator labels.
    motion = jnp.abs(target - frames[:, :-1])
    return jnp.mean((1.0 + 8.0 * motion) * jnp.square(predicted - target))
