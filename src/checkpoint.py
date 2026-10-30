"""Atomic, restartable checkpoints for deterministic JAX training runs."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def save(path, params, first, second, step, seed, batch_size,
         architecture="recurrent", curriculum="baseline", replay_fraction=0.10, total_updates=0):
    """Commit complete optimizer state atomically; readers see only full files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for prefix, state in (("p", params), ("m", first), ("v", second)):
        arrays.update({f"{prefix}_{name}": np.asarray(value)
                       for name, value in state.items()})
    arrays.update(step=np.int64(step), seed=np.int64(seed), batch_size=np.int64(batch_size),
                  architecture=np.asarray(architecture), curriculum=np.asarray(curriculum),
                  replay_fraction=np.float64(replay_fraction), total_updates=np.int64(total_updates))
    temp = path.with_suffix(".part")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load(path, seed, batch_size,
         architecture="recurrent", curriculum="baseline", replay_fraction=0.10, total_updates=0):
    """Return (params, first, second, step), rejecting incompatible runs."""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["seed"]) != seed or int(archive["batch_size"]) != batch_size:
            raise ValueError("Checkpoint seed/batch-size differs; start a new run")
        if (str(archive["architecture"]) != architecture
                or str(archive["curriculum"]) != curriculum
                or abs(float(archive["replay_fraction"]) - replay_fraction) > 1e-12
                or int(archive["total_updates"]) != total_updates):
            raise ValueError("Checkpoint architecture/curriculum/horizon differs")
        names = ("local", "readout", "bias")
        states = [{name: jnp.asarray(archive[f"{prefix}_{name}"]) for name in names}
                  for prefix in ("p", "m", "v")]
        step = int(archive["step"])
    return (*states, step)
