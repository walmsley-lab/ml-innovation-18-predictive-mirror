"""A shared local recurrent rule over arbitrary connected geometry.

No complete geometry-specific lookup weights, object identities, future sensor
values, latent displacement, velocity, or privileged simulator state enter the
model. Geometry is specified only by a normalized one-hop adjacency matrix.
"""
import jax
import jax.numpy as jnp


def init(key, channels=12):
    if channels < 2:
        raise ValueError("Require at least two recurrent channels")
    a, b = jax.random.split(key)
    # Features: observed anchor 2, innovation 2, mask 1, outgoing force 2,
    # neighbor anchored displacement 2, hidden and neighbor hidden each C.
    return {"local": .09*jax.random.normal(a, (9+2*channels, channels)),
            "bias": jnp.zeros((channels,), jnp.float32),
            "readout": .02*jax.random.normal(b, (channels, 2))}


def advance(params, state, observation, mask, force, graph, use_messages=True):
    """Apply one shared node rule to infer what follows the current contact."""
    hidden, prior = state
    anchor = mask * observation + (1-mask) * prior
    innovation = mask * (observation - prior)
    messages = graph @ hidden if use_messages else jnp.zeros_like(hidden)
    neighbor_anchor = graph @ anchor if use_messages else jnp.zeros_like(anchor)
    features = jnp.concatenate((anchor, innovation, mask, force, neighbor_anchor,
                                hidden, messages), axis=-1)
    hidden = .7*hidden + .3*jnp.tanh(features @ params["local"]+params["bias"])
    prediction = anchor + .16*jnp.tanh(hidden @ params["readout"])
    return (hidden, prediction), prediction


def predict(params, observed, mask, forces, graph, use_messages=True):
    """Issue one-step predictions before receiving the next observation."""
    if observed.ndim != 3 or observed.shape[-1] != 2:
        raise ValueError("Expected observed[T+1,N,2]")
    time, nodes, _ = observed.shape
    if (time < 2 or mask.shape != (time, nodes, 1) or
            forces.shape != (time-1, nodes, 2) or graph.shape != (nodes, nodes)):
        raise ValueError("Incompatible graph/sensor/action sequence")
    state = (jnp.zeros((nodes, params["bias"].shape[0]), jnp.float32),
             jnp.zeros((nodes, 2), jnp.float32))

    def transition(state, sensors):
        perception, visibility, action = sensors
        return advance(params, state, perception, visibility, action,
                       graph, use_messages)
    _, predictions = jax.lax.scan(
        transition, state, (observed[:-1], mask[:-1], forces))
    return predictions


def loss(params, observed, mask, forces, graph, use_messages=True):
    issued = predict(params, observed, mask, forces, graph, use_messages)
    squared = jnp.sum((issued - observed[1:])**2, axis=-1, keepdims=True)
    # Train only on observable sensors; no privileged hidden-state supervision.
    return jnp.sum(squared * mask[1:]) / jnp.maximum(1., mask[1:].sum()*2)
