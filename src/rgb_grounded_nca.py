"""Anti-collapse variants of the predictive RGB NCA.

The baseline ``predictive_rgb_nca`` commits
``future = clip(image + 0.20*tanh(hidden @ rgb_readout + bias), 0, 1)``. On the
sparse motion world this additive-residual head, fed back on itself during
open-loop dreaming, drives every pixel to the lower clip rail: the rollout
reaches an input-independent all-black fixed point. These variants keep the same
cellular core (``predictive_rgb_nca.cell_update``/``observe``) and investigate
two of the levers for avoiding that collapse, with ``mode="control"``
reproducing the baseline exactly:

* **output dynamics** (``output="blend"``) -- a convex alpha-composite
  ``future = (1-gate)*image + gate*color`` with ``gate, color`` in ``(0,1)``.
  Being a convex blend of two in-range quantities it cannot accumulate past the
  rail, so the all-black fixed point is no longer absorbing.
* **foreground-sensitive loss** (``foreground_weight>0``) -- extra loss weight on
  true foreground (bright object) pixels, so predicting background is costly.

Balanced scene sampling is the third lever; it belongs to the data sampler, not
the model (see ``experiments.training_budget_sweep``).
"""
import jax
import jax.numpy as jnp

from src import predictive_rgb_nca as nca

OUTPUT_MODES = ("residual", "blend")
FOREGROUND_LEVEL = 0.30


def init(key, channels=8, output="residual"):
    """Baseline parameters, plus a per-pixel gate head for ``output='blend'``."""
    if output not in OUTPUT_MODES:
        raise ValueError("Unknown output mode")
    params = nca.init(key, channels)
    if output == "blend":
        gate_key = jax.random.fold_in(key, 7)
        params = {
            **params,
            # Small negative gate bias -> start near persistence (gate ~ 0.4),
            # not saturated, so early training can move either way.
            "gate_readout": 0.05 * jax.random.normal(gate_key, (channels, 1)),
            "gate_bias": jnp.full((1,), -0.4, jnp.float32),
        }
    return params


def dream(params, state, ticks=2, output="residual"):
    hidden, image = state
    quiet = jnp.zeros_like(image)

    def iterate(h, _):
        return nca.cell_update(params, h, image, quiet), None

    hidden, _ = jax.lax.scan(iterate, hidden, None, length=ticks)
    if output == "residual":
        correction = 0.20 * jnp.tanh(hidden @ params["rgb_readout"]
                                     + params["rgb_bias"])
        future = jnp.clip(image + correction, 0.0, 1.0)
    else:  # blend: convex alpha-composite, bounded in [0,1] without clipping.
        color = jax.nn.sigmoid(hidden @ params["rgb_readout"] + params["rgb_bias"])
        gate = jax.nn.sigmoid(hidden @ params["gate_readout"] + params["gate_bias"])
        future = (1.0 - gate) * image + gate * color
    return (hidden, future), future


def rollout_and_activity(params, observed_context, horizon, ticks=2,
                         output="residual"):
    if (output not in OUTPUT_MODES or horizon < 1 or ticks < 1
            or observed_context.ndim != 5 or observed_context.shape[-1] != 3):
        raise ValueError("Need RGB [B,C,H,W,3], valid horizon, ticks and output")
    batch, context, height, width, _ = observed_context.shape
    if context < 2:
        raise ValueError("Require at least two observed context frames")
    state = nca.initial_state(params, batch, height, width)

    def absorb_and_dream(s, frame):
        corrected = nca.observe(params, s, frame, ticks)
        predicted_state, _ = dream(params, corrected, ticks, output)
        return predicted_state, None

    state, _ = jax.lax.scan(
        absorb_and_dream, state,
        jnp.swapaxes(observed_context[:, :-1], 0, 1))
    state = nca.observe(params, state, observed_context[:, -1], ticks)

    def predict(s, _):
        s, frame = dream(params, s, ticks, output)
        activity = jnp.mean(jnp.abs(s[0]), axis=-1, keepdims=True)
        return s, (frame, activity)

    _, (rgb, activity) = jax.lax.scan(predict, state, None, length=horizon)
    return jnp.swapaxes(rgb, 0, 1), jnp.swapaxes(activity, 0, 1)


def rollout(params, observed_context, horizon, ticks=2, output="residual"):
    return rollout_and_activity(params, observed_context, horizon, ticks, output)[0]


def loss(params, examples, context=6, horizon=5, ticks=2, output="residual",
         foreground_weight=0.0, foreground_level=FOREGROUND_LEVEL):
    """Motion- and (optionally) foreground-weighted open-loop prediction loss."""
    if foreground_weight < 0:
        raise ValueError("foreground_weight must be nonnegative")
    predictions = rollout(params, examples[:, :context], horizon, ticks, output)
    target = examples[:, context:context + horizon]
    motion = jnp.mean(jnp.abs(target - examples[:, context - 1:context]),
                      axis=-1, keepdims=True)
    foreground = (jnp.max(target, axis=-1, keepdims=True)
                  > foreground_level).astype(jnp.float32)
    weights = 1.0 + 8.0 * motion + foreground_weight * foreground
    return jnp.mean(weights * jnp.square(predictions - target))
