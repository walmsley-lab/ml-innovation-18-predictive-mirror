"""Compare synthetic pretraining against matched scratch webcam models.

Same architecture, initial parameters, real-data minibatches and fine-tuning
steps. Extra pretraining compute is explicit, not hidden in the comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import procedural_video, video_residual
from train import adamw
from video_train import clips, windows


def evaluate(params, heldout):
    actions = jnp.zeros((len(heldout), heldout.shape[1] - 1, 1), jnp.float32)
    predicted = np.asarray(jax.vmap(video_residual.predict, in_axes=(None, 0, 0))(
        params, jnp.asarray(heldout), actions))
    target, observed = heldout[:, 1:], heldout[:, :-1]
    motion = np.abs(target - observed) > 0.03
    sq = np.square(predicted - target)
    return {
        "mse": float(sq.mean()),
        "motion_region_mse": float(sq[motion].mean()) if motion.any() else None,
        "motion_pixels": int(motion.sum()),
    }


def run(input_path, output, seed=41, pretrain_steps=100, finetune_steps=100,
        batch_size=8, length=8):
    if pretrain_steps < 0 or finetune_steps < 1 or batch_size < 1 or length < 2:
        raise ValueError("Invalid training schedule")
    with np.load(input_path, allow_pickle=False) as archive:
        frames = archive["frames"].astype(np.float32)
    if (frames.ndim != 4 or frames.shape[1] != frames.shape[2]
            or frames.shape[-1] != 1 or frames.shape[1] < 16):
        raise ValueError("Expected [T,size,size,1] grayscale recording, size >=16")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("Invalid pixel values")
    train_idx, val_idx = windows(frames, length)
    initial = video_residual.init(jax.random.PRNGKey(seed))
    zeros = lambda p: jax.tree.map(jnp.zeros_like, p)

    @jax.jit
    def update(p, m, v, samples, i):
        actions = jnp.zeros((samples.shape[0], length, 1), jnp.float32)
        loss, grads = jax.value_and_grad(video_residual.loss)(p, samples, actions)
        p, m, v = adamw(p, grads, m, v, i)
        return p, m, v, loss

    pretrained, pm, pv = initial, zeros(initial), zeros(initial)
    for step in range(1, pretrain_steps + 1):
        stage = min(3, (step - 1) * 4 // pretrain_steps)
        seeds = seed * 1000000 + step * batch_size + np.arange(batch_size)
        synthetic = procedural_video.batch(seeds, [stage] * batch_size,
                                           steps=length, size=frames.shape[1])
        pretrained, pm, pv, _ = update(pretrained, pm, pv, jnp.asarray(synthetic), step)

    # Reset optimizer moments for fine-tuning, so both arms use the same state.
    scratch, sm, sv = initial, zeros(initial), zeros(initial)
    pm, pv = zeros(pretrained), zeros(pretrained)
    rng = np.random.default_rng(seed + 77)
    for step in range(1, finetune_steps + 1):
        chosen = rng.choice(train_idx, size=batch_size, replace=True)
        samples = jnp.asarray(clips(frames, chosen, length))
        pretrained, pm, pv, _ = update(pretrained, pm, pv, samples, step)
        scratch, sm, sv, _ = update(scratch, sm, sv, samples, step)

    heldout = clips(frames, val_idx, length)
    trained, control = evaluate(pretrained, heldout), evaluate(scratch, heldout)
    persistence = float(np.mean((heldout[:, :-1] - heldout[:, 1:]) ** 2))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "pretrained.npz", **jax.tree.map(np.asarray, pretrained))
    np.savez_compressed(output / "scratch.npz", **jax.tree.map(np.asarray, scratch))
    result = {
        "architecture": "video_residual",
        "stages": ["one_disc", "two_discs", "wall_occlusion", "camera_motion"],
        "pretrain_steps": pretrain_steps, "finetune_steps_per_arm": finetune_steps,
        "extra_compute_for_pretraining": pretrain_steps,
        "seed": seed, "validation_windows": len(heldout),
        "pretrained": trained, "scratch": control, "persistence_mse": persistence,
        "transfer_gain_mse": control["mse"] - trained["mse"],
        "recording_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "note": "Exploratory one-recording holdout; pretraining uses additional compute and does not establish transfer causation",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="recordings/webcam.npz")
    p.add_argument("--out", default="runs/synthetic-transfer")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--pretrain-steps", type=int, default=100)
    p.add_argument("--finetune-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    print(json.dumps(run(args.input, args.out, args.seed, args.pretrain_steps,
                         args.finetune_steps, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
