"""Training orders with exactly matched example multisets and exposure budgets."""

import numpy as np


def schedule(seed, updates, batch_size, mode="staged", replay_fraction=0.10):
    """Return (seeds, stages): [updates,batch_size]; stages 0..3."""
    if mode not in ("staged", "shuffled"):
        raise ValueError(f"Unsupported curriculum: {mode}")
    if not (0.0 <= replay_fraction <= 1.0):
        raise ValueError("replay_fraction must be in [0,1]")
    rng = np.random.default_rng(seed + 707)
    stages = np.empty((updates, batch_size), np.int8)
    for step in range(updates):
        current = min(3, step * 4 // updates)
        stages[step] = current
        if current > 0:
            replay = rng.random(batch_size) < replay_fraction
            stages[step, replay] = rng.integers(0, current, size=int(replay.sum()))
    seeds = seed * 1_000_000 + np.arange(updates * batch_size).reshape(updates, batch_size)
    if mode == "shuffled":
        # Reorder identical (seed, stage) pairs; no altered example count or difficulty mix.
        ordering = rng.permutation(updates * batch_size)
        seeds = seeds.ravel()[ordering].reshape(updates, batch_size)
        stages = stages.ravel()[ordering].reshape(updates, batch_size)
    return seeds, stages
