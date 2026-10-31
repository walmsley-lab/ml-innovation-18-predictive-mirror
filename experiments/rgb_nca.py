"""RGB capture + independently trained predictive neural cellular automaton."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca
from train import adamw
from video_train import clips, windows


def preprocess_color(bgr, size, cv2):
    """Center-cropped BGR camera image -> RGB floats. Never convert to GREY."""
    if bgr.ndim != 3 or bgr.shape[-1] != 3:
        raise ValueError("Expected 3-channel camera image; check RGB camera selection")
    height, width = bgr.shape[:2]
    side = min(height, width)
    crop = bgr[(height-side)//2:(height-side)//2+side,
               (width-side)//2:(width-side)//2+side]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def capture(path="recordings/rgb-webcam.npz", camera=0, seconds=60, fps=10, size=32):
    """Record local RGB frames; never uploads data or opens the IR stream on purpose."""
    import cv2
    if seconds <= 0 or fps <= 0 or not 16 <= size <= 64:
        raise ValueError("seconds/fps must be positive and size 16..64")
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera} unavailable (check fuser /dev/video{camera})")
    # Advisory camera format; camera driver can ignore. The user must verify RGB.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frames, stamps = [], []
    start = time.monotonic()
    deadline = start
    try:
        while time.monotonic() - start < seconds:
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            ok, raw = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped yielding frames")
            frames.append(preprocess_color(raw, size, cv2))
            stamps.append(time.monotonic_ns())
            deadline = max(deadline + 1 / fps, time.monotonic())
    finally:
        cap.release()
    if len(frames) < 100:
        raise ValueError("Need at least 100 frames for training and validation")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, frames=np.stack(frames),
                        timestamps=np.asarray(stamps, np.int64),
                        camera_index=np.int32(camera))
    return len(frames)


def read_recording(path):
    with np.load(path, allow_pickle=False) as recording:
        frames = recording["frames"].astype(np.float32)
        stamps = recording["timestamps"] if "timestamps" in recording.files else None
    if (frames.ndim != 4 or frames.shape[-1] != 3
            or frames.shape[1] != frames.shape[2] or frames.shape[1] < 16):
        raise ValueError("Expected RGB recording [T,size,size,3]; grayscale inputs cannot train this model")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("Invalid RGB samples")
    return frames, stamps


def evaluate(params, heldout, context, horizon, ticks):
    preds = np.asarray(nca.rollout(params, jnp.asarray(heldout[:, :context]), horizon, ticks))
    actual = heldout[:, context:context + horizon]
    last = heldout[:, context-1:context]
    prior = heldout[:, context-2:context-1]
    results = {}
    for lead in sorted(set((1, 3, 5, 10, horizon)) & set(range(1, horizon+1))):
        truth = actual[:, lead-1]
        pred = preds[:, lead-1]
        hold = last[:, 0]
        velocity = np.clip(hold + lead * (hold-prior[:, 0]), 0, 1)
        changed = np.mean(np.abs(truth-hold), axis=-1) > .03
        squared = np.mean(np.square(pred-truth), axis=-1)
        baseline = np.mean(np.square(hold-truth), axis=-1)
        results[str(lead)] = {
            "model_mse": float(squared.mean()),
            "persistence_mse": float(baseline.mean()),
            "pixel_extrapolation_mse": float(np.square(velocity-truth).mean()),
            "motion_region_model_mse": float(squared[changed].mean()) if changed.any() else None,
            "motion_region_persistence_mse": float(baseline[changed].mean()) if changed.any() else None,
            "motion_pixels": int(changed.sum()),
        }
    return results


def fit(input_path="recordings/rgb-webcam.npz", output="runs/rgb-nca",
        seed=41, steps=100, batch_size=4, context=6, horizon=5, ticks=2,
        channels=12):
    if steps < 1 or batch_size < 1 or context < 2 or horizon < 1 or ticks < 1:
        raise ValueError("Invalid steps, batch, context, horizon or NCA ticks")
    frames, stamps = read_recording(input_path)
    length = context + horizon - 1
    train_idx, val_idx = windows(frames, length)
    params = nca.init(jax.random.PRNGKey(seed), channels)
    moments = jax.tree.map(jnp.zeros_like, params)
    variances = jax.tree.map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed)
    heldout = clips(frames, val_idx, length)

    @jax.jit
    def update(p, m, v, examples, step):
        error, grads = jax.value_and_grad(
            lambda weights: nca.loss(weights, examples, context, horizon, ticks))(p)
        # Basic global-norm clipping protects against occasional motion spikes.
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
        grads = jax.tree.map(lambda g: g / jnp.maximum(1.0, norm), grads)
        p, m, v = adamw(p, grads, m, v, step)
        return p, m, v, error

    # Select a checkpoint on held-out validation; this is NOT an unbiased test set.
    base_score = evaluate(params, heldout, context, horizon, ticks)
    best_metric = float(np.mean([base_score[str(lead)]["model_mse"]
                                 for lead in (1, horizon)]))
    best_params = params
    best_step = 0
    rng_steps = []
    for i in range(1, steps + 1):
        sample = jnp.asarray(clips(frames, rng.choice(train_idx, size=batch_size,
                                                    replace=True), length))
        params, moments, variances, error = update(params, moments, variances, sample, i)
        if i % max(1, steps // 5) == 0 or i == steps:
            score = evaluate(params, heldout, context, horizon, ticks)
            metric = float(np.mean([score[str(lead)]["model_mse"]
                                    for lead in (1, horizon)]))
            if metric < best_metric:
                best_metric, best_params, best_step = metric, params, i
            rng_steps.append({"step": i, "train_loss": float(error),
                              "validation_selection_mse": metric})
            print(f"step={i} training_loss={float(error):.6f} val={metric:.6f}", flush=True)

    results = evaluate(best_params, heldout, context, horizon, ticks)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "parameters.npz", **jax.tree.map(np.asarray, best_params))
    report = {
        "architecture": "predictive_rgb_nca", "channels": channels,
        "nca_ticks_per_observation_and_dream": ticks,
        "context_frames": context, "forecast_horizon_frames": horizon,
        "training_steps": steps, "best_validation_step": best_step,
        "seed": seed, "validation_windows": len(val_idx),
        "recording_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "horizons": results, "training_history": rng_steps,
        "note": "RGB 2D locally gated NCA, explicit prediction error, unobserved dreams. "
                "No prior grayscale weights transferred. Chronological validation chosen for "
                "checkpoint selection; scores are NOT independent test estimates.",
    }
    if stamps is not None and len(stamps)>1:
        report["median_capture_interval_ms"] = float(np.median(np.diff(stamps).astype(np.float64)) / 1e6)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("capture")
    record.add_argument("--camera", type=int, default=0)
    record.add_argument("--seconds", type=int, default=60)
    record.add_argument("--fps", type=float, default=10)
    record.add_argument("--size", type=int, default=32)
    record.add_argument("--out", default="recordings/rgb-webcam.npz")
    train = sub.add_parser("fit")
    train.add_argument("--input", default="recordings/rgb-webcam.npz")
    train.add_argument("--out", default="runs/rgb-nca")
    train.add_argument("--seed", type=int, default=41)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--context", type=int, default=6)
    train.add_argument("--horizon", type=int, default=5)
    train.add_argument("--ticks", type=int, default=2)
    train.add_argument("--channels", type=int, default=12)
    args = parser.parse_args()
    if args.command == "capture":
        print(f"Captured {capture(args.out, args.camera, args.seconds, args.fps, args.size)} RGB frames")
    else:
        print(json.dumps(fit(args.input, args.out, args.seed, args.steps,
                             args.batch_size, args.context, args.horizon,
                             args.ticks, args.channels), indent=2))


if __name__ == "__main__":
    main()
