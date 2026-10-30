"""Capture local webcam frames or train a tiny persistence-residual predictor."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import video_residual
from train import adamw


def capture(path, camera=0, seconds=120, fps=10, size=32):
    """Save only local gray video. No uploads or remote telemetry."""
    import cv2
    from mirror import preprocess
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera} unavailable")
    frames, timestamps = [], []
    start = time.monotonic()
    deadline = start
    try:
        while time.monotonic() - start < seconds:
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            success, raw = cap.read()
            if not success:
                break
            frames.append(preprocess(raw, size, cv2))
            timestamps.append(time.monotonic_ns())
            deadline += 1.0 / fps
    finally:
        cap.release()
    if len(frames) < 40:
        raise ValueError("Need at least 40 frames for training and a held-out check")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=np.stack(frames), timestamps=np.array(timestamps, dtype=np.int64))
    return len(frames)


def windows(frames, length=8):
    """Disjoint chronological train/validation regions with an embargo."""
    n = len(frames)
    if n < max(80, 10 * length):
        raise ValueError("Recording is too short for a separated validation region")
    first_end = int(0.70 * n)
    validation_start = int(0.80 * n)
    train_starts = np.arange(0, first_end - length)
    validation_starts = np.arange(validation_start, n - length, length + 1)
    if not len(train_starts) or not len(validation_starts):
        raise ValueError("No valid temporal training/validation windows")
    return train_starts, validation_starts


def clips(frames, starts, length):
    return np.stack([frames[int(i):int(i) + length + 1] for i in starts])


def fit(input_path, output, seed=41, steps=500, batch_size=8, length=8):
    """Train on recorded images; score temporally separated held-out windows."""
    with np.load(input_path, allow_pickle=False) as archive:
        frames = archive["frames"].astype(np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 1 or frames.shape[1] != frames.shape[2]:
        raise ValueError("Expected [T,size,size,1] grayscale frames")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("Frames must be finite floats in [0,1]")
    train_idx, val_idx = windows(frames, length)
    rng = np.random.default_rng(seed)
    params = video_residual.init(jax.random.PRNGKey(seed))
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)

    @jax.jit
    def update(p, m, v, batch, i):
        actions = jnp.zeros((batch.shape[0], length, 1), jnp.float32)
        error, gradients = jax.value_and_grad(video_residual.loss)(p, batch, actions)
        p, m, v = adamw(p, gradients, m, v, i)
        return p, m, v, error

    for step in range(1, steps + 1):
        indices = rng.choice(train_idx, size=batch_size, replace=True)
        batch = clips(frames, indices, length)
        params, first, second, error = update(params, first, second, jnp.asarray(batch), step)
        if step % max(steps // 5, 1) == 0:
            print(f"step={step} training_loss={float(error):.6f}", flush=True)

    heldout = clips(frames, val_idx, length)
    actions = jnp.zeros((len(heldout), length, 1), jnp.float32)
    predictions = np.asarray(jax.vmap(video_residual.predict, in_axes=(None, 0, 0))(
        params, jnp.asarray(heldout), actions))
    actual, baseline = heldout[:, 1:], heldout[:, :-1]
    mse = float(np.mean((predictions - actual) ** 2))
    baseline_mse = float(np.mean((baseline - actual) ** 2))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "parameters.npz", **jax.tree.map(np.asarray, params))
    metrics = {
        "architecture": "video_residual", "training_steps": steps,
        "frame_size": int(frames.shape[1]), "validation_windows": len(heldout),
        "validation_mse": mse, "validation_persistence_mse": baseline_mse,
        "beats_persistence": bool(mse < baseline_mse),
        "warning": "single-recording temporal holdout is demo validation, not independent scientific evidence",
        "recording_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--seconds", type=int, default=120)
    c.add_argument("--fps", type=float, default=10)
    c.add_argument("--size", type=int, default=32)
    c.add_argument("--out", default="recordings/webcam.npz")
    t = sub.add_parser("fit")
    t.add_argument("--input", default="recordings/webcam.npz")
    t.add_argument("--out", default="runs/webcam")
    t.add_argument("--seed", type=int, default=41)
    t.add_argument("--steps", type=int, default=500)
    t.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.command == "capture":
        if args.seconds <= 0 or args.fps <= 0 or not 8 <= args.size <= 64:
            parser.error("seconds/fps must be positive, size between 8 and 64")
        print(f"Captured {capture(args.out, args.camera, args.seconds, args.fps, args.size)} frames locally")
    else:
        if args.steps < 1 or args.batch_size < 1:
            parser.error("Steps and batch size must be positive")
        print(json.dumps(fit(args.input, args.out, args.seed, args.steps, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
