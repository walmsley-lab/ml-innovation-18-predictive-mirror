"""Equal-budget webcam test: ordinary recurrence vs oscillators, K=0 vs K>0."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import oscillatory_video, video_residual
from train import adamw
from video_train import clips, windows


def scores(params, heldout, mode, coupling):
    actions = jnp.zeros((len(heldout), heldout.shape[1] - 1, 1), jnp.float32)
    if mode == "baseline":
        predictor = jax.vmap(video_residual.predict, in_axes=(None, 0, 0))
        predicted = np.asarray(predictor(params, jnp.asarray(heldout), actions))
    else:
        predictor = jax.vmap(oscillatory_video.predict, in_axes=(None, 0, 0, None))
        predicted = np.asarray(predictor(params, jnp.asarray(heldout), actions, coupling))
    truth = heldout[:, 1:]
    previous = heldout[:, :-1]
    mask = np.abs(truth - previous) > 0.03
    square = np.square(predicted - truth)
    return {"mse": float(square.mean()),
            "motion_region_mse": float(square[mask].mean()) if mask.any() else None,
            "motion_pixels": int(mask.sum())}


def run(input_path, output, seed=41, steps=100, batch_size=8, coupling=0.25, length=8):
    if steps < 1 or batch_size < 1 or coupling <= 0:
        raise ValueError("steps/batch_size/coupling must be positive")
    with np.load(input_path, allow_pickle=False) as file:
        frames = file["frames"].astype(np.float32)
    if (frames.ndim != 4 or frames.shape[-1] != 1
            or frames.shape[1] != frames.shape[2]):
        raise ValueError("Expected grayscale square frames [T,H,W,1]")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("Invalid pixel values")
    train_idx, val_idx = windows(frames, length)
    initial = video_residual.init(jax.random.PRNGKey(seed))
    zeros = lambda p: jax.tree.map(jnp.zeros_like, p)

    def make_update(mode, k):
        @jax.jit
        def update(p, m, v, batch, i):
            action = jnp.zeros((batch.shape[0], length, 1), jnp.float32)
            if mode == "baseline":
                objective = lambda weights: video_residual.loss(weights, batch, action)
            else:
                objective = lambda weights: oscillatory_video.loss(weights, batch, action, k)
            error, grads = jax.value_and_grad(objective)(p)
            p, m, v = adamw(p, grads, m, v, i)
            return p, m, v, error
        return update

    names = ("baseline", "oscillator_uncoupled", "oscillator_coupled")
    settings = (("baseline", 0.0), ("oscillator", 0.0), ("oscillator", coupling))
    weights = {name: initial for name in names}
    moments = {name: (zeros(initial), zeros(initial)) for name in names}
    updates = {name: make_update(*setting) for name, setting in zip(names, settings)}
    rng = np.random.default_rng(seed + 77)
    for i in range(1, steps + 1):
        chosen = rng.choice(train_idx, size=batch_size, replace=True)
        batch = jnp.asarray(clips(frames, chosen, length))
        for name in names:
            m, v = moments[name]
            weights[name], m, v, _ = updates[name](weights[name], m, v, batch, i)
            moments[name] = m, v

    heldout = clips(frames, val_idx, length)
    result = {
        "architecture": "video_residual_plus_fixed_phase",
        "training_steps_each": steps,
        "trainable_parameters_matched": True,
        "extra_state": "1 phase channel per pixel in oscillator arms",
        "seed": seed, "coupling": coupling, "validation_windows": len(heldout),
        "persistence_mse": float(np.mean(np.square(heldout[:, :-1] - heldout[:, 1:]))),
        "arms": {name: scores(weights[name], heldout, *setting)
                 for name, setting in zip(names, settings)},
        "recording_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "note": "Single-recording temporal holdout; no demonstrated benefit until replicated on independent videos",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        np.savez_compressed(output / (name + ".npz"), **jax.tree.map(np.asarray, weights[name]))
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="recordings/webcam.npz")
    p.add_argument("--out", default="runs/phase-ablation")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--coupling", type=float, default=0.25)
    args = p.parse_args()
    print(json.dumps(run(args.input, args.out, args.seed, args.steps,
                         args.batch_size, args.coupling), indent=2))


if __name__ == "__main__":
    main()
