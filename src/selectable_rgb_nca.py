"""Independent RGB NCA with explicit spatial selection, not the fixed 3x3 model.

All modes share cellular rule, weights and training interface. References
are external pointers, not learned object identities or inferred geometry.
"""
import jax
import jax.numpy as jnp

MODES = ("local", "relative", "reference")
OFFSETS = ((-2, 0), (2, 0), (0, -2), (0, 2))


def reference_map(height, width):
    """Deterministic nonlocal pointers (diagonal mirror): NOT learned."""
    y, x = jnp.indices((height, width))
    return jnp.stack((height - 1 - y, width - 1 - x), axis=-1)


def select(features, mode="local", offsets=OFFSETS, references=None):
    """Select contexts [B,H,W,C] with one explicitly specified rule."""
    if features.ndim != 4 or mode not in MODES:
        raise ValueError("Requires [B,H,W,C] and local/relative/reference mode")
    batch, height, width, channels = features.shape
    if mode == "local":
        return jax.lax.reduce_window(features, 0.0, jax.lax.add,
                                     (1, 3, 3, 1), (1, 1, 1, 1), "SAME") / 9.0
    if mode == "relative":
        if not offsets:
            raise ValueError("At least one relative offset required")
        pad = max(max(abs(int(y)), abs(int(x))) for y, x in offsets)
        padded = jnp.pad(features, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        samples = [padded[:, pad + y:pad + y + height,
                          pad + x:pad + x + width, :]
                   for y, x in offsets]
        return sum(samples) / len(samples)
    if references is None or references.shape != (height, width, 2):
        raise ValueError("Reference mode requires an external [H,W,2] pointer map")
    y, x = references[..., 0], references[..., 1]
    valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    index = jnp.clip(y, 0, height-1)*width + jnp.clip(x, 0, width-1)
    flat = features.reshape(batch, height*width, channels)
    sampled = jnp.take(flat, index.reshape(-1), axis=1).reshape(features.shape)
    return jnp.where(valid[None, ..., None], sampled, 0.0)


def init(key, channels=12):
    if channels < 1:
        raise ValueError("channels must be positive")
    key, _ = jax.random.split(key)
    features = channels + 6
    return {"projection": 0.05 * jax.random.normal(key, (2*features, 2*channels)),
            "bias": jnp.zeros((2*channels,), jnp.float32),
            "readout": jnp.zeros((channels, 3), jnp.float32),
            "rgb_bias": jnp.zeros((3,), jnp.float32)}


def cell_update(params, hidden, image, surprise, mode="local", references=None):
    own = jnp.concatenate((hidden, image, surprise), axis=-1)
    chosen = select(own, mode, references=references)
    mixed = jnp.concatenate((own, chosen), axis=-1) @ params["projection"] + params["bias"]
    gate, proposal = jnp.split(mixed, 2, axis=-1)
    return hidden + 0.125 * jax.nn.sigmoid(gate) * jnp.tanh(proposal)


def observe(params, state, frame, ticks=2, mode="local", references=None):
    hidden, predicted = state
    error = frame - predicted
    def tick(h, _):
        return cell_update(params, h, frame, error, mode, references), None
    hidden, _ = jax.lax.scan(tick, hidden, None, length=ticks)
    return hidden, frame


def dream(params, state, ticks=2, mode="local", references=None):
    hidden, image = state
    quiet = jnp.zeros_like(image)
    def tick(h, _):
        return cell_update(params, h, image, quiet, mode, references), None
    hidden, _ = jax.lax.scan(tick, hidden, None, length=ticks)
    predicted = jnp.clip(
        image + 0.2*jnp.tanh(hidden @ params["readout"] + params["rgb_bias"]), 0, 1)
    return (hidden, predicted), predicted


def rollout(params, context, horizon=5, ticks=2, mode="local", references=None):
    """Open-loop RGB dreams from observed context; never reads future frames."""
    if (context.ndim != 5 or context.shape[-1] != 3 or context.shape[1] < 2
            or ticks < 1 or horizon < 1 or mode not in MODES):
        raise ValueError("Requires RGB context [B,C,H,W,3] and positive ticks/horizon")
    batch, _, height, width, _ = context.shape
    if mode == "reference" and references is None:
        raise ValueError("Explicit reference pointer map required")
    state = (jnp.zeros((batch, height, width, params["readout"].shape[0]), jnp.float32),
             jnp.zeros((batch, height, width, 3), jnp.float32))
    def warm(s, frame):
        s = observe(params, s, frame, ticks, mode, references)
        s, _ = dream(params, s, ticks, mode, references)
        return s, None
    state, _ = jax.lax.scan(warm, state, jnp.swapaxes(context[:, :-1], 0, 1))
    state = observe(params, state, context[:, -1], ticks, mode, references)
    def forecast(s, _):
        return dream(params, s, ticks, mode, references)
    _, predictions = jax.lax.scan(forecast, state, None, length=horizon)
    return jnp.swapaxes(predictions, 0, 1)


def loss(params, episodes, context=6, horizon=5, ticks=2, mode="local", references=None):
    predicted = rollout(params, episodes[:, :context], horizon, ticks, mode, references)
    truth = episodes[:, context:context+horizon]
    motion = jnp.mean(jnp.abs(truth - episodes[:, context-1:context]), axis=-1,
                      keepdims=True)
    return jnp.mean((1 + 8*motion)*jnp.square(predicted-truth))
