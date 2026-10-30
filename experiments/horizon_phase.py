"""Standalone long-horizon predictive mirror experiment.

Train plain recurrence, uncoupled phase, and coupled phase to predict H frames
autoregressively. Future observations are ONLY labels, never rollout inputs.
Does not modify the original one-step model or its training code.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import video_residual, oscillatory_video
from train import adamw
from video_train import clips, windows


ARMS = ("plain", "phase_uncoupled", "phase_coupled")


def rollout_one(params, observed, horizon, arm, coupling):
    """Predict [H,Y,X,1] from observed [C,Y,X,1], no future observations."""
    if horizon < 1 or len(observed) < 2 or arm not in ARMS:
        raise ValueError("Need horizon >=1, context >=2, valid arm")
    spatial = observed.shape[1:3]
    zero_action = jnp.zeros((1, 1), jnp.float32)
    if arm == "plain":
        state = jnp.zeros((1, *spatial, params["local"].shape[-1]), jnp.float32)
        def advance(state, frame):
            return video_residual.step(params, state, frame, zero_action)
    else:
        state = oscillatory_video.initial_state(params, (1, *spatial))
        strength = 0.0 if arm == "phase_uncoupled" else coupling
        def advance(state, frame):
            return oscillatory_video.step(params, state, frame, zero_action, strength)

    def warm(state, frame):
        state, _ = advance(state, frame[None])
        return state, None

    state, _ = jax.lax.scan(warm, state, observed[:-1])

    def predict_next(carry, _):
        state, frame = carry
        state, prediction = advance(state, frame)
        return (state, prediction), prediction[0]

    (_, _), predictions = jax.lax.scan(
        predict_next, (state, observed[-1:]), None, length=horizon)
    return predictions


def rollout_batch(params, context, horizon, arm, coupling):
    return jax.vmap(
        lambda observed: rollout_one(params, observed, horizon, arm, coupling)
    )(context)


def horizon_loss(params, examples, context, horizon, arm, coupling):
    """Multi-step loss through predicted frames; no teacher forcing in rollout."""
    history, target = examples[:, :context], examples[:, context:context+horizon]
    output = rollout_batch(params, history, horizon, arm, coupling)
    # Amplify moving pixels and longer horizons; future truth is a label only.
    motion = jnp.abs(target - history[:, -1:])
    weights = 1.0 + 8.0 * motion
    lead_weights = (1.0 + jnp.arange(horizon, dtype=jnp.float32) / horizon)
    lead_weights = lead_weights.reshape(1, horizon, 1, 1, 1)
    return jnp.mean(weights * lead_weights * jnp.square(output - target))


def score_arm(params, heldout, context, horizon, arm, coupling, leads):
    history = jnp.asarray(heldout[:, :context])
    predicted = np.asarray(rollout_batch(params, history, horizon, arm, coupling))
    truth = heldout[:, context:context+horizon]
    last = heldout[:, context-1]
    previous = heldout[:, context-2]
    result = {}
    for step in leads:
        actual = truth[:, step-1]
        forecast = predicted[:, step-1]
        pers = last
        # Pixelwise finite-difference extrapolation, not optical flow.
        linear = np.clip(last + step * (last - previous), 0, 1)
        mask = np.abs(actual - last) > 0.03
        sq = np.square(forecast - actual)
        result[str(step)] = {
            "model_mse": float(sq.mean()),
            "persistence_mse": float(np.square(pers - actual).mean()),
            "pixel_difference_extrapolation_mse": float(np.square(linear - actual).mean()),
            "motion_region_model_mse": float(sq[mask].mean()) if mask.any() else None,
            "motion_region_persistence_mse": (
                float(np.square(pers - actual)[mask].mean()) if mask.any() else None),
            "motion_pixels": int(mask.sum()),
        }
    return result


def run(input_path, output, seed=41, steps=100, batch_size=8,
        context=8, horizon=10, coupling=0.25):
    if steps < 1 or batch_size < 1 or context < 2 or horizon < 1 or coupling <= 0:
        raise ValueError("Positive steps/batch/horizon/coupling; context>=2 required")
    with np.load(input_path, allow_pickle=False) as source:
        frames = source["frames"].astype(np.float32)
        timestamps = source["timestamps"] if "timestamps" in source.files else None
    if (frames.ndim != 4 or frames.shape[-1] != 1
            or frames.shape[1] != frames.shape[2] or frames.shape[1] < 8):
        raise ValueError("Expected grayscale square video [T,H,W,1]")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("Pixel values must be finite in [0,1]")
    length = context + horizon - 1
    train_idx, val_idx = windows(frames, length)
    initial = video_residual.init(jax.random.PRNGKey(seed))
    parameters = {name: initial for name in ARMS}
    zeros = lambda p: jax.tree.map(jnp.zeros_like, p)
    first = {name: zeros(initial) for name in ARMS}
    second = {name: zeros(initial) for name in ARMS}

    def updater(arm):
        @jax.jit
        def update(p, m, v, batch, i):
            objective = lambda weights: horizon_loss(
                weights, batch, context, horizon, arm, coupling)
            error, gradient = jax.value_and_grad(objective)(p)
            p, m, v = adamw(p, gradient, m, v, i)
            return p, m, v, error
        return update

    updates = {name: updater(name) for name in ARMS}
    rng = np.random.default_rng(seed + 77)
    history = []
    for i in range(1, steps + 1):
        indices = rng.choice(train_idx, size=batch_size, replace=True)
        sample = jnp.asarray(clips(frames, indices, length))
        errors = {}
        for name in ARMS:
            p, m, v, error = updates[name](
                parameters[name], first[name], second[name], sample, i)
            parameters[name], first[name], second[name] = p, m, v
            errors[name] = float(error)
        if i == steps or i % max(1, steps // 5) == 0:
            history.append({"step": i, "training_loss": errors})
            print(f"step={i} " + " ".join(f"{a}={errors[a]:.6f}" for a in ARMS), flush=True)

    heldout = clips(frames, val_idx, length)
    leads = tuple(sorted(set((1, 3, 5, 10, horizon)) & set(range(1, horizon+1))))
    scores = {name: score_arm(parameters[name], heldout, context, horizon,
                              name, coupling, leads) for name in ARMS}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ARMS:
        np.savez_compressed(out / f"{name}.npz",
                            **jax.tree.map(np.asarray, parameters[name]))
    intervals_ms = (np.diff(timestamps).astype(np.float64) / 1e6
                    if timestamps is not None and len(timestamps)>1 else None)
    report = {
        "experiment": "separate_autoregressive_multi_horizon_phase",
        "arms": scores, "leads_in_frames": leads,
        "history_frames": context, "forecast_frames": horizon,
        "seed": seed, "training_steps_per_arm": steps,
        "trainable_parameters_matched": True,
        "oscillator_extra_hidden_state": "one phase channel per pixel in phase arms",
        "coupling": coupling, "training_history": history,
        "validation_windows": len(val_idx),
        "validation": "chronological final 20% with 10% embargo; nonoverlapping clips",
        "median_capture_interval_ms": (
            float(np.median(intervals_ms)) if intervals_ms is not None else None),
        "recording_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "limitations": ("Single-recording holdout, no independent replication; "
                        "multi-horizon model is NOT a 3D model or object tracker. "
                        "Pixelwise linear baseline is not optical flow. "
                        "Performance differences cannot be attributed to phase synchrony "
                        "without repeated seeds, latency checks and ablations."),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="recordings/webcam.npz")
    p.add_argument("--out", default="runs/horizon-phase")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--coupling", type=float, default=0.25)
    args = p.parse_args()
    print(json.dumps(run(args.input, args.out, args.seed, args.steps,
                         args.batch_size, args.context, args.horizon,
                         args.coupling), indent=2))


if __name__ == "__main__":
    main()
