"""Neuromatch-inspired probes for the existing RGB predictive NCA.

No learned performance claims: the CLI evaluates one frozen checkpoint.
Future target images are used only after forecasts are issued.
"""
import numpy as np


def sparse_observe(previous, observed, threshold=0.02, max_fraction=0.25):
    """Assimilate observed RGB changes at selected pixels; count actual updates."""
    a, b = np.asarray(previous, float), np.asarray(observed, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("Expected matching [H,W,3] images")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite observation")
    if threshold < 0 or not 0 <= max_fraction <= 1:
        raise ValueError("Invalid threshold or update budget")
    surprise = np.abs(a - b).mean(axis=-1).reshape(-1)
    eligible = np.flatnonzero(surprise > threshold)
    budget = int(np.ceil(max_fraction * surprise.size))
    chosen = eligible[np.argsort(-surprise[eligible], kind="stable")[:budget]]
    out = a.copy().reshape(-1, 3)
    out[chosen] = b.reshape(-1, 3)[chosen]
    return out.reshape(a.shape), {
        "updated_pixels": int(len(chosen)), "total_pixels": int(surprise.size),
        "observed_rgb_reads": int(3 * len(chosen)),
        "measured_energy_joules": None,
    }


def normalize_rgb(image, radius=2, eps=0.05):
    """Fixed local divisive contrast normalization; no temporal information."""
    x = np.asarray(image, float)
    if x.ndim != 3 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite [H,W,3] image")
    if radius < 0 or eps <= 0:
        raise ValueError("Invalid normalization window")
    luminance = x.mean(axis=-1)
    padded = np.pad(luminance, radius, mode="edge")
    side = 2 * radius + 1
    local = sum(padded[i:i+x.shape[0], j:j+x.shape[1]]
                for i in range(side) for j in range(side)) / (side * side)
    return np.clip(x * (0.5 + eps) / (local[..., None] + eps), 0, 1)


def velocity_cka(first, second):
    """Centered-kernel alignment of temporal state *increments*.

    A small dynamical-similarity proxy, NOT the full DSA procedure.
    Preserves time alignment; invariant to orthogonal channel rotations.
    """
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or len(a) < 3:
        raise ValueError("Need aligned [time>=3,features] trajectories")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite states")
    a, b = np.diff(a, axis=0), np.diff(b, axis=0)
    a -= a.mean(axis=0, keepdims=True)
    b -= b.mean(axis=0, keepdims=True)
    ka, kb = a @ a.T, b @ b.T
    denom = np.linalg.norm(ka) * np.linalg.norm(kb)
    return float(np.sum(ka * kb) / denom) if denom > 1e-15 else 0.0


def observed_trajectory(params, frames, ticks=1):
    """Collect mean channel activity after each *observed* RGB frame."""
    from src import predictive_rgb_nca as nca
    import jax.numpy as jnp
    x = np.asarray(frames)
    if x.ndim != 4 or x.shape[-1] != 3 or ticks < 1:
        raise ValueError("Expected [T,H,W,3] video and positive ticks")
    state = nca.initial_state(params, 1, x.shape[1], x.shape[2])
    trajectory = []
    for frame in x:
        state = nca.observe(params, state, jnp.asarray(frame[None]), ticks)
        trajectory.append(np.asarray(state[0]).mean(axis=(0, 1, 2)))
        state, _ = nca.dream(params, state, ticks)
    return np.stack(trajectory)


def targeted_channel_lesions(params, context, truth, horizon=3, ticks=1,
                             fraction=0.25, seed=9):
    """Ablate high-energy state channels vs count- and energy-matched random.

    Selection uses observed context only; evaluator truth only scores forecasts.
    A frozen checkpoint is required for meaningful interpretations.
    """
    from src import predictive_rgb_nca as nca, rgb_hidden_interventions as hi
    import jax.numpy as jnp
    x, y = np.asarray(context), np.asarray(truth)
    if x.ndim != 5 or y.shape != (x.shape[0], horizon, *x.shape[2:]) or x.shape[-1] != 3:
        raise ValueError("Expected context [B,C,H,W,3] and future [B,horizon,H,W,3]")
    if horizon < 1 or ticks < 1 or not 0 < fraction <= 1:
        raise ValueError("Invalid horizon, ticks or lesion fraction")
    state = hi.warm_state(params, jnp.asarray(x), ticks)
    energy = np.square(np.asarray(state[0])).sum(axis=(0, 1, 2))
    channels = len(energy)
    n = max(1, round(channels * fraction))
    if n >= channels:
        raise ValueError("A distinct random lesion needs at least one spared channel")
    targeted = np.argsort(-energy, kind="stable")[:n]
    rng = np.random.default_rng(seed)
    # An identical group cannot be a null intervention. For tiny networks,
    # enumerate all groups; random sample for larger channel counts.
    from itertools import combinations
    if channels <= 12:
        candidates = [np.asarray(c, int) for c in combinations(range(channels), n)
                      if set(c) != set(targeted)]
    else:
        candidates = [c for c in (rng.choice(channels, n, replace=False)
                      for _ in range(512)) if set(c) != set(targeted)]
    if not candidates:
        raise ValueError("Could not construct a distinct lesion null")
    random_channels = min(candidates, key=lambda c: abs(energy[c].sum() - energy[targeted].sum()))
    def run(indices):
        hidden, image = state
        if len(indices):
            mask = np.ones(channels, np.float32)
            mask[np.asarray(indices, int)] = 0
            hidden = hidden * jnp.asarray(mask)
        result = []
        for _ in range(horizon):
            (hidden, image), predicted = nca.dream(params, (hidden, image), ticks)
            result.append(np.asarray(predicted))
        return float(np.square(np.stack(result, axis=1) - y).mean())
    return {
        "intact_mse": run([]),
        "targeted_mse": run(targeted),
        "matched_random_mse": run(random_channels),
        "targeted_channels": targeted.tolist(),
        "random_channels": random_channels.tolist(),
        "targeted_removed_energy": float(energy[targeted].sum()),
        "random_removed_energy": float(energy[random_channels].sum()),
        "selection_uses_future_truth": False,
    }
