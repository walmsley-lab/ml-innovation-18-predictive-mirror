"""Video-residual predictor with a coupled, fixed-rule phase-oscillator sheet.

Phase adds hidden state, not learnable parameters. Coupling can be set to zero
to retain an otherwise identical oscillating control.
"""
import jax
import jax.numpy as jnp

from src import video_residual
from src.model import conv


def init(key):
    return video_residual.init(key)


def initial_state(params, shape):
    batch, height, width = shape
    return (jnp.zeros((batch, height, width, params["local"].shape[-1]), jnp.float32),
            jnp.zeros((batch, height, width, 1), jnp.float32))


def neighbor_alignment(phase):
    """Mean of sin(theta_neighbor - theta_center) for 8 neighboring pixels."""
    kernel = jnp.ones((3, 3, 1, 1), dtype=phase.dtype) / 8
    kernel = kernel.at[1, 1, 0, 0].set(0.0)
    sine, cosine = jnp.sin(phase), jnp.cos(phase)
    sin_neighbors = conv(sine, kernel)
    cos_neighbors = conv(cosine, kernel)
    return cosine * sin_neighbors - sine * cos_neighbors


def step(params, state, frame, action, coupling=0.25):
    hidden, phase = state
    if coupling < 0:
        raise ValueError("coupling must be nonnegative")
    action_image = jnp.ones_like(frame) * action.reshape(-1, 1, 1, 1)
    proposal = jnp.tanh(conv(jnp.concatenate((frame, action_image, hidden), axis=-1),
                             params["local"]))
    # A fixed natural frequency, observation-driven phase perturbation, and
    # local Kuramoto-like coupling. No hidden object labels or future frames.
    next_phase = phase + 0.35 + 0.10 * jnp.mean(proposal, axis=-1, keepdims=True)
    next_phase = next_phase + coupling * neighbor_alignment(phase)
    gain = 1.0 + 0.4 * jnp.sin(next_phase)
    updated = 0.7 * hidden + 0.3 * gain * proposal
    delta = 0.25 * jnp.tanh(conv(updated, params["readout"]) + params["bias"])
    return (updated, next_phase), jnp.clip(frame + delta, 0.0, 1.0)


def predict(params, frames, actions, coupling=0.25):
    state = initial_state(params, (1, frames.shape[1], frames.shape[2]))

    def advance(current, pair):
        frame, action = pair
        updated, predicted = step(params, current, frame[None], action[None], coupling)
        return updated, predicted[0]

    _, predictions = jax.lax.scan(advance, state, (frames[:-1], actions))
    return predictions


def loss(params, frames, actions, coupling=0.25):
    predicted = jax.vmap(predict, in_axes=(None, 0, 0, None))(
        params, frames, actions, coupling)
    target, previous = frames[:, 1:], frames[:, :-1]
    motion = jnp.abs(target - previous)
    return jnp.mean((1.0 + 8.0 * motion) * jnp.square(predicted - target))
