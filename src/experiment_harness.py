"""Shared scaffolding for experiment runners.

These helpers factor out the four patterns that were copy-pasted across most
`experiments/` runners (see docs/EXPERIMENT_LEDGER.md cleanup backlog):

- the grad-norm-clipped AdamW update step,
- zero-initialised Adam moments,
- the "evaluate at these leads" horizon idiom,
- frozen-checkpoint loading with a provenance hash, and metrics writing.

They are behaviour-preserving: `clipped_adamw_update` reproduces exactly the
`norm = sqrt(sum g^2); grad /= max(norm, 1); adamw(...)` step the runners wrote
inline, and `adamw` keeps its existing default rate (0.002).
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from train import adamw


def init_moments(params):
    """Zero first/second Adam moments matching `params`' tree shape."""
    zeros = jax.tree.map(jnp.zeros_like, params)
    return zeros, jax.tree.map(jnp.zeros_like, params)


def clipped_adamw_update(loss, rate=0.002):
    """Build a jitted training step: global grad-norm clip to 1, then AdamW.

    `loss(params, *batch) -> scalar`. The returned
    `update(params, first, second, batch, step) -> (params, first, second, loss)`
    is the exact step the runners previously inlined; define `loss` as a closure
    over any static arguments (predictor, context, horizon, ...).
    """
    @jax.jit
    def update(params, first, second, batch, step):
        value, grad = jax.value_and_grad(lambda q: loss(q, *batch))(params)
        norm = jnp.sqrt(sum((g * g).sum() for g in jax.tree.leaves(grad)))
        grad = jax.tree.map(lambda g: g / jnp.maximum(norm, 1.0), grad)
        params, first, second = adamw(params, grad, first, second, step, rate)
        return params, first, second, value
    return update


def leads(horizon, candidates=(1, 3, 5, 10)):
    """The forecast leads to score: `horizon` plus each candidate within it.

    Reproduces the `tuple(sorted({horizon, *(l for l in (1,3,5,10) if l<=horizon)}))`
    idiom repeated across the RGB runners.
    """
    return tuple(sorted({horizon, *(lead for lead in candidates if lead <= horizon)}))


def load_checkpoint(path):
    """Load a frozen `.npz` checkpoint as a jax array dict, with its SHA256.

    Returns `(params, sha256_hex)`. `allow_pickle=False` keeps loading to plain
    arrays only, matching the runners' provenance-conscious loaders.
    """
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as handle:
        params = {key: jnp.asarray(handle[key]) for key in handle.files}
    return params, digest


def write_metrics(path, obj):
    """Write `obj` as pretty JSON (trailing newline), creating parent dirs.

    Returns the `Path` written, or None when `path` is falsy (the runners use a
    None/empty path to mean "do not persist").
    """
    if not path:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    return path
