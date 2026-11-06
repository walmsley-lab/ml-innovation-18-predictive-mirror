"""JAX implementation of the local developmental sheet.

The equations intentionally mirror DevelopmentalSheet in the NumPy reference
so a single-step numerical-equivalence test can guard the GPU path.
"""
import jax
import jax.numpy as jnp


def pool(frame, sheet):
    h, w, c = frame.shape
    if c != 3 or h % sheet or w % sheet:
        raise ValueError("RGB frame must be divisible by sheet")
    return frame.reshape(sheet, h // sheet, sheet, w // sheet, 3).mean((1, 3))


def init(sheet=4, channels=7, seed=11, plasticity=.04):
    if sheet < 2 or sheet % 2 or channels < 7:
        raise ValueError("Need an even sheet and at least seven channels")
    key = jax.random.PRNGKey(seed)
    phase_key, connection_key = jax.random.split(key)
    return {
        "phase": jax.random.uniform(phase_key, (sheet, sheet),
                                    minval=-jnp.pi, maxval=jnp.pi),
        "hidden": jnp.zeros((sheet, sheet, channels), jnp.float32),
        "trace": jnp.zeros((sheet, sheet, channels), jnp.float32),
        "homeostasis": jnp.zeros((channels,), jnp.float32),
        "connections": .01 * jax.random.normal(
            connection_key, (sheet, sheet, channels, 3)),
        "previous": jnp.zeros((sheet, sheet, 3), jnp.float32),
        "forecast": jnp.zeros((sheet, sheet, 3), jnp.float32),
        "plasticity": jnp.asarray(plasticity, jnp.float32),
    }


def step(state, frame, action=(0., 0.), plastic=True):
    sheet = state["phase"].shape[0]
    fast = pool(frame, sheet)
    slow = fast.reshape(sheet // 2, 2, sheet // 2, 2, 3).mean((1, 3))
    slow = jnp.repeat(jnp.repeat(slow, 2, 0), 2, 1)
    wave = jnp.sin(state["phase"])[..., None]
    x = jnp.concatenate((fast, slow, wave), axis=-1)
    retinal = fast
    surprise = jnp.mean(jnp.abs(retinal - state["forecast"]))
    action_signal = .02 * action[0] + .01 * action[1]
    trace = .82 * state["trace"] + .18 * jnp.tanh(x * surprise)
    pre = jnp.tanh(x + .25 * state["hidden"] + .08 * trace +
                    state["homeostasis"] + action_signal)
    post = jnp.tanh(pre + .05 * jnp.roll(pre, 1, 0) + .05 * jnp.roll(pre, 1, 1))
    correlation = pre[..., :, None] * post[..., None, :3]
    correlation = correlation - correlation.mean(axis=2, keepdims=True)
    rate = state["plasticity"] * jnp.asarray(plastic, dtype=jnp.float32)
    connections = state["connections"] + rate * jnp.clip(surprise, 0., 1.) * correlation
    connections = connections * (1. - state["plasticity"] * .02)
    activity = jnp.mean(jnp.abs(post), axis=(0, 1))
    homeostasis = jnp.clip(state["homeostasis"] + .02 * (.12 - activity), -.25, .25)
    connections = connections / jnp.maximum(1., jnp.linalg.norm(connections, axis=2, keepdims=True))
    phase = state["phase"] + .25 + .08 * jnp.roll(state["phase"], 1, 0) - .08 * state["phase"]
    phase = phase + .01 * action[0] + .005 * action[1]
    delta = jnp.tanh(jnp.einsum("hwc,hwcr->hwr", post, connections))
    prediction = jnp.clip(state["previous"] + .18 * delta, 0., 1.)
    return {"phase": phase, "hidden": post, "trace": trace,
            "homeostasis": homeostasis, "connections": connections,
            "previous": retinal, "forecast": prediction,
            "plasticity": state["plasticity"]}, prediction


def forecast(state):
    return state["forecast"]
