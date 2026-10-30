"""Matched-history convolutional predictor with no persistent recurrent state."""

import jax
import jax.numpy as jnp

from src.model import conv


HISTORY = 12


def init(key, channels=3):
    a, b = jax.random.split(key)
    return {
        "local": jax.random.normal(a, (3, 3, HISTORY * 2, channels)) * 0.08,
        "readout": jax.random.normal(b, (1, 1, channels, 1)) * 0.08,
        "bias": jnp.array(-2.5),
    }


def predict(params, frames, actions):
    """Each forecast gets up to 12 past frames/actions, never future inputs."""
    h, w = frames.shape[1:3]
    images = jnp.zeros((HISTORY, h, w, 1), jnp.float32)
    commands = jnp.zeros((HISTORY,), jnp.float32)

    def move(carry, input_pair):
        images, commands = carry
        frame, action = input_pair
        images = jnp.concatenate((images[1:], frame[None]), axis=0)
        commands = jnp.concatenate((commands[1:], action.reshape(1)), axis=0)
        image_channels = images.transpose(1, 2, 0, 3).reshape(h, w, HISTORY)
        action_channels = jnp.broadcast_to(commands, (h, w, HISTORY))
        observation = jnp.concatenate((image_channels, action_channels), axis=-1)[None]
        hidden = jnp.tanh(conv(observation, params["local"]))
        forecast = jax.nn.sigmoid(conv(hidden, params["readout"]) + params["bias"])
        return (images, commands), forecast[0]

    _, outputs = jax.lax.scan(move, (images, commands), (frames[:-1], actions))
    return outputs


def loss(params, frames, actions):
    forecast = jax.vmap(predict, in_axes=(None, 0, 0))(params, frames, actions)
    return jnp.mean((forecast - frames[:, 1:]) ** 2)
