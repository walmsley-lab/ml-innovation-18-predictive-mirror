"""Small deterministic RGB predictor with an explicit global R^d recurrent state.

This first experiment is a point predictor, not a posterior over explanations.
The RGB decoder adds a coarse appearance change to the last image. Latent
coordinates have no prescribed physical meaning and receive no geometry labels.
"""

import jax
import jax.numpy as jnp

from src.model import conv


def pool(frames, grid=8):
    b, h, w, c = frames.shape
    if h % grid or w % grid:
        raise ValueError("Image dimensions must be divisible by latent input grid")
    return frames.reshape(b, grid, h//grid, grid, w//grid, c).mean((2, 4))


def init(key, latent_dim=64):
    if latent_dim < 2:
        raise ValueError("Latent dimension must be at least two")
    a, b, c, d = jax.random.split(key, 4)
    return {
        "perception": jax.random.normal(a, (3, 3, 3, 4)) * .1,
        "encoder": jax.random.normal(b, (8*8*4, latent_dim)) / jnp.sqrt(8*8*4),
        "input": jax.random.normal(c, (latent_dim, 3*latent_dim)) / jnp.sqrt(latent_dim),
        "recurrent": jax.random.normal(d, (latent_dim, 3*latent_dim)) / jnp.sqrt(latent_dim),
        "bias": jnp.zeros((3*latent_dim,), jnp.float32),
        "readout": jnp.zeros((latent_dim, 8*8*3), jnp.float32),
        "rgb_bias": jnp.zeros((8*8*3,), jnp.float32),
    }


def recurrent(params, state, encoded):
    xr, xu, xn = jnp.split(encoded @ params["input"] + params["bias"], 3, axis=-1)
    hr, hu, hn = jnp.split(state @ params["recurrent"], 3, axis=-1)
    reset = jax.nn.sigmoid(xr + hr)
    keep = jax.nn.sigmoid(xu + hu + 1.)
    candidate = jnp.tanh(xn + reset * hn)
    return keep * state + (1 - keep) * candidate


def encode_history(params, context):
    if (context.ndim != 5 or context.shape[-1] != 3 or context.shape[1] < 2
            or min(context.shape) < 1):
        raise ValueError("Need nonempty RGB [B,T,H,W,3], T>=2")
    state = jnp.zeros((context.shape[0], params["readout"].shape[0]), jnp.float32)

    def observe(state, frame):
        visual = pool(jnp.tanh(conv(frame, params["perception"])))
        encoded = jnp.tanh(visual.reshape(len(frame), -1) @ params["encoder"])
        state = recurrent(params, state, encoded)
        return state, None

    state, _ = jax.lax.scan(observe, state, jnp.swapaxes(context, 0, 1))
    return state


def dream(params, state, image):
    state = recurrent(params, state, jnp.zeros_like(state))
    correction = .2 * jnp.tanh(state @ params["readout"] + params["rgb_bias"])
    correction = correction.reshape(len(image), 8, 8, 3)
    correction = jax.image.resize(correction, image.shape, method="linear")
    return state, jnp.clip(image + correction, 0., 1.)


def rollout(params, context, horizon=6):
    if horizon < 1:
        raise ValueError("Need a positive forecast horizon")
    state = encode_history(params, context)

    def advance(carry, _):
        state, image = dream(params, *carry)
        return (state, image), image

    _, predicted = jax.lax.scan(advance, (state, context[:, -1]), None, length=horizon)
    return jnp.swapaxes(predicted, 0, 1)
