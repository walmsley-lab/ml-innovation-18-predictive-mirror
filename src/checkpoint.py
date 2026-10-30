"""Atomic, restartable checkpoints for deterministic JAX training runs."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def save(path, params, first, second, step, seed, batch_size):
    """Commit complete optimizer state atomically; readers see only full files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for prefix, state in (("p", params), ("m", first), ("v", second)):
        arrays.update({f"{prefix}_{name}": np.asarray(value)
                       for name, value in state.items()})
    arrays.update(step=np.int64(step), seed=np.int64(seed), batch_size=np.int64(batch_size))
    temp = path.with_suffix(".part")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load(path, seed, batch_size):
    """Return (params, first, second, step), rejecting incompatible runs."""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["seed"]) != seed or int(archive["batch_size"]) != batch_size:
            raise ValueError("Checkpoint seed/batch-size differs; start a new run")
        names = ("local", "readout", "bias")
        states = [{name: jnp.asarray(archive[f"{prefix}_{name}"]) for name in names}
                  for prefix in ("p", "m", "v")]
        step = int(archive["step"])
    return (*states, step)
