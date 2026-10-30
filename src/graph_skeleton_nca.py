"""Graph neural cellular automaton over supplied 2D skeleton joints.

Each node holds its own learned state, receives neighbor messages, and produces
next-frame joint positions. The graph is GIVEN, not discovered from pixels.
"""
import jax
import jax.numpy as jnp

# 12-node simplified stick figure; no depth, rendering, or camera model.
JOINTS = ("pelvis", "neck", "left_shoulder", "left_elbow", "left_wrist",
          "right_shoulder", "right_elbow", "right_wrist", "left_knee",
          "left_ankle", "right_knee", "right_ankle")
EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
         (0, 8), (8, 9), (0, 10), (10, 11))


def adjacency(edges=EDGES, nodes=len(JOINTS)):
    """Undirected, degree-normalized one-hop graph; no self edges."""
    graph = jnp.zeros((nodes, nodes), jnp.float32)
    for i, j in edges:
        if not (0 <= i < nodes and 0 <= j < nodes) or i == j:
            raise ValueError("Invalid skeleton edge")
        graph = graph.at[i, j].set(1.0)
        graph = graph.at[j, i].set(1.0)
    return graph / jnp.maximum(graph.sum(axis=1, keepdims=True), 1.0)


def init(key, channels=16):
    if channels < 1:
        raise ValueError("channels must be positive")
    local_key, readout_key = jax.random.split(key)
    return {"local": 0.07 * jax.random.normal(local_key, (5 + 2 * channels, channels)),
            "bias": jnp.zeros((channels,), jnp.float32),
            "readout": jnp.zeros((channels, 2), jnp.float32)}


def initial_state(batch_size, nodes, channels):
    return (jnp.zeros((batch_size, nodes, channels), jnp.float32),
            jnp.zeros((batch_size, nodes, 2), jnp.float32))


def step(params, state, observed, visibility, graph, use_messages=True):
    """Advance from visible joints only; hidden coordinates never enter inputs.

    observed [B,N,2] must already zero out occluded joints.
    visibility [B,N,1] is binary; graph [N,N] known topology.
    """
    hidden, prior = state
    mask = visibility.astype(jnp.float32)
    anchor = mask * observed + (1.0 - mask) * prior
    messages = jnp.einsum("ij,bjc->bic", graph, hidden) if use_messages else jnp.zeros_like(hidden)
    features = jnp.concatenate((anchor, mask * observed, mask, hidden, messages), axis=-1)
    updated = 0.7 * hidden + 0.3 * jnp.tanh(features @ params["local"] + params["bias"])
    delta = 0.12 * jnp.tanh(updated @ params["readout"])
    forecast = jnp.clip(anchor + delta, 0.0, 1.0)
    return (updated, forecast), forecast


def predict(params, observed, visibility, graph, use_messages=True):
    """[B,T+1,N,2] -> [B,T,N,2]; never read frame T while predicting it."""
    batch_size, _, nodes, _ = observed.shape
    state = initial_state(batch_size, nodes, params["readout"].shape[0])

    def advance(current, pair):
        frames, masks = pair
        return step(params, current, frames, masks, graph, use_messages)

    _, forecasts = jax.lax.scan(
        advance, state,
        (jnp.swapaxes(observed[:, :-1], 0, 1),
         jnp.swapaxes(visibility[:, :-1], 0, 1)))
    return jnp.swapaxes(forecasts, 0, 1)


def loss(params, observed, visibility, graph, use_messages=True):
    prediction = predict(params, observed, visibility, graph, use_messages)
    # Future visible joints only: ground-truth occluded positions are NEVER labels.
    target, mask = observed[:, 1:], visibility[:, 1:]
    squared = jnp.sum(jnp.square(prediction - target), axis=-1, keepdims=True)
    return jnp.sum(mask * squared) / jnp.maximum(mask.sum(), 1.0)
