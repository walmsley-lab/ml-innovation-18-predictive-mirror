"""Train the fixed-rule recurrent field on tiny synthetic sequences."""

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.model import init, loss, predict
from src.checkpoint import load as load_checkpoint, save as save_checkpoint
from src.world import batch, sequence, curriculum_batch
from src.curriculum import schedule


def adamw(params, grads, first, second, step_number, rate=0.002):
    """Small explicit AdamW; no optimizer framework needed for the experiment."""
    first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, grads)
    second = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, second, grads)
    params = jax.tree.map(
        lambda p, m, v: p * (1 - rate * 0.0001)
        - rate * (m / (1 - 0.9 ** step_number))
        / (jnp.sqrt(v / (1 - 0.999 ** step_number)) + 1e-8),
        params, first, second)
    return params, first, second


def train(seed=42, steps=120, batch_size=8, output="runs/pilot", require_gpu=False,
          checkpoint_every=25, resume=False, curriculum="baseline", replay_fraction=0.10):
    devices = jax.devices()
    if require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU requested but not available: {devices}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    params = init(jax.random.PRNGKey(seed))
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    checkpoint_path = out / "checkpoint.npz"
    start_step = 0
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
        params, first, second, start_step = load_checkpoint(
            checkpoint_path, seed, batch_size, "recurrent", "baseline", 0.10, 0)
        if start_step > steps:
            raise ValueError("Checkpoint is beyond requested total steps")

    @jax.jit
    def update(p, m, v, x, a, i):
        error, gradients = jax.value_and_grad(loss)(p, x, a)
        p, m, v = adamw(p, gradients, m, v, i)
        return p, m, v, error

    if curriculum != "baseline":
        training_seeds, training_stages = schedule(seed, steps, batch_size, curriculum, replay_fraction)
    history = []
    for n in range(start_step + 1, steps + 1):
        # Unique simulation seeds per update; fixed seed makes experiments reproducible.
        if curriculum == "baseline":
            seeds = seed * 1_000_000 + n * batch_size + np.arange(batch_size)
            x, a = batch(seeds, steps=12)
        else:
            x, a = curriculum_batch(training_seeds[n - 1], training_stages[n - 1], steps=12)
        params, first, second, err = update(params, first, second, jnp.asarray(x), jnp.asarray(a), n)
        if checkpoint_every > 0 and (n % checkpoint_every == 0 or n == steps):
            save_checkpoint(checkpoint_path, params, first, second, n, seed, batch_size,
                            "recurrent", "baseline", 0.10, 0)
        if n == start_step + 1 or n % max(1, steps // 10) == 0:
            history.append({"step": n, "loss": float(err)})
            print(f"step={n} loss={float(err):.5f}", flush=True)

    # Untouched, disjoint test trajectories and visible baselines.
    # Ground-truth masks are held by the evaluator only, never used to train.
    test_episodes = [sequence(int(s), steps=12) for s in 10_000_000 + np.arange(64)]
    test_frames = np.stack([episode[0] for episode in test_episodes])
    test_actions = np.stack([episode[1] for episode in test_episodes])
    visible_masks = np.stack([episode[2]["visible_mask"][1:] for episode in test_episodes])
    y = np.asarray(jax.vmap(predict, in_axes=(None, 0, 0))(
        params, jnp.asarray(test_frames), jnp.asarray(test_actions)))
    truth = test_frames[:, 1:]
    persistence = test_frames[:, :-1]
    weighted = 1.0 + 8.0 * visible_masks
    metrics = {
        "test_mse": float(np.mean((y - truth) ** 2)),
        "persistence_mse": float(np.mean((persistence - truth) ** 2)),
        "test_weighted_mse": float(np.mean(weighted * (y - truth) ** 2)),
        "persistence_weighted_mse": float(np.mean(weighted * (persistence - truth) ** 2)),
        "test_object_pixel_mse": float(np.mean(((y - truth) ** 2)[visible_masks])) if visible_masks.any() else None,
        "device": [str(d) for d in devices],
        "seed": seed, "steps": steps, "batch_size": batch_size,
        "curriculum": curriculum, "replay_fraction": replay_fraction,
        "status": "pilot_only_not_evidence_of_emergence",
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(out / "parameters.npz", **jax.tree.map(np.asarray, params))
    np.savez_compressed(out / "example.npz", actual=test_frames[0], prediction=y[0], camera_actions=test_actions[0])
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--steps", type=int, default=int(os.getenv("STEPS", "120")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "8")))
    parser.add_argument("--output", default=os.getenv("OUTPUT", "runs/pilot"))
    parser.add_argument("--curriculum", choices=("baseline", "staged", "shuffled"), default=os.getenv("CURRICULUM", "baseline"))
    parser.add_argument("--replay-fraction", type=float, default=float(os.getenv("REPLAY_FRACTION", "0.10")))
    parser.add_argument("--checkpoint-every", type=int, default=int(os.getenv("CHECKPOINT_EVERY", "25")))
    parser.add_argument("--resume", action="store_true", default=os.getenv("RESUME", "0") == "1")
    parser.add_argument("--require-gpu", action="store_true", default=os.getenv("REQUIRE_GPU", "0") == "1")
    args = parser.parse_args()
    train(args.seed, args.steps, args.batch_size, args.output, args.require_gpu,
          args.checkpoint_every, args.resume, args.curriculum, args.replay_fraction)


if __name__ == "__main__":
    main()
