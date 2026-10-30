"""Two recurrent perceptual sheets trained without cross-sheet communication."""

import jax
import jax.numpy as jnp

from src import multisheet


def init(key):
    return multisheet.init(key)


def predict(params, frames, actions):
    return multisheet.predict(params, frames, actions, feedback_strength=0.0)


def loss(params, frames, actions, loss_mode="plain"):
    forecast = jax.vmap(predict, in_axes=(None, 0, 0))(params, frames, actions)
    target = frames[:, 1:]
    error = (forecast - target) ** 2
    return jnp.mean((1.0 + 32.0 * (target > 0.9)) * error) if loss_mode == "contrast" else jnp.mean(error)
