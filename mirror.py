"""Causal webcam mirror: predict before retrieving the next observation."""

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src import model as recurrent, multisheet


def preprocess(bgr, size, cv2):
    """Center-crop, convert BGR to grayscale, area-resize, normalize to [0,1]."""
    height, width = bgr.shape[:2]
    side = min(height, width)
    crop = bgr[(height - side) // 2:(height - side) // 2 + side,
               (width - side) // 2:(width - side) // 2 + side]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return (cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
            .astype(np.float32) / 255.0)[..., None]


def initial_state(architecture, size):
    if architecture == "recurrent":
        return jnp.zeros((1, size, size, 8), jnp.float32)
    if architecture == "multisheet":
        return (jnp.zeros((1, size, size, multisheet.FAST_CHANNELS), jnp.float32),
                jnp.zeros((1, size, size, multisheet.SLOW_CHANNELS), jnp.float32))
    raise ValueError("The live camera adapter supports recurrent and multisheet only")


def predict_next(params, architecture, state, observed):
    """Only the current observation and known action=unknown/zero are inputs."""
    x = jnp.asarray(observed[None], dtype=jnp.float32)
    a = jnp.zeros((1, 1), jnp.float32)  # unknown camera action placeholder
    network = {"recurrent": recurrent, "multisheet": multisheet}[architecture]
    next_state, forecast = network.step(params, state, x, a)
    return next_state, np.asarray(forecast[0])


def forecast_pairs(observations, state, predictor, clock=time.monotonic_ns):
    """Commit each forecast before requesting the corresponding next frame."""
    stream = iter(observations)
    try:
        current, current_arrived = next(stream)
    except StopIteration:
        return
    previous = None
    while True:
        started = clock()
        state, prediction = predictor(state, current)
        committed = clock()
        try:
            actual, arrived = next(stream)
        except StopIteration:
            return
        if arrived <= committed:
            raise RuntimeError("Captured frame timestamp predates prediction commit")
        extrapolated = np.clip(current + (current - previous), 0, 1) if previous is not None else current
        yield {
            "actual": actual, "predicted": prediction, "last_frame": current,
            "extrapolated": extrapolated,
            "prediction_committed_ns": committed,
            "frame_arrived_ns": arrived,
            "inference_ms": (committed - started) / 1e6,
            "elapsed_to_frame_ms": (arrived - committed) / 1e6,
        }
        previous, current, current_arrived = current, actual, arrived


def camera_frames(cap, size, fps, cv2):
    interval = 1.0 / fps
    deadline = time.monotonic()
    while True:
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        success, bgr = cap.read()
        arrived = time.monotonic_ns()
        if not success:
            return
        yield preprocess(bgr, size, cv2), arrived
        deadline = max(deadline + interval, time.monotonic())


def panel(result, scale, cv2):
    frames = [result["actual"], result["predicted"],
              np.abs(result["predicted"] - result["actual"])]
    labels = ("Observed", "Predicted", "Absolute error")
    panels = []
    for label, array in zip(labels, frames):
        gray = (255 * np.clip(array[..., 0], 0, 1)).astype(np.uint8)
        large = cv2.resize(gray, (scale, scale), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(large, cv2.COLOR_GRAY2BGR)
        cv2.putText(rgb, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 240, 230), 1)
        panels.append(rgb)
    return np.hstack(panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", required=True, help="Previously trained parameters.npz")
    parser.add_argument("--architecture", choices=("recurrent", "multisheet"), default="recurrent")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--scale", type=int, default=240)
    parser.add_argument("--out", default="runs/webcam-metrics.json")
    args = parser.parse_args()
    if args.size < 8 or args.size > 128 or args.fps <= 0:
        parser.error("size must be 8..128 and fps must be positive")
    try:
        import cv2
    except ImportError:
        parser.error("Install opencv-python for the local webcam demo")
    with np.load(args.parameters, allow_pickle=False) as archive:
        params = {name: jnp.asarray(archive[name]) for name in archive.files}
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # request only; drivers may ignore
    state = initial_state(args.architecture, args.size)
    report = {"architecture": args.architecture, "pixels": args.size,
              "nominal_fps": args.fps, "action_channel": "unknown_zero_placeholder",
              "model_source": args.parameters, "frames": []}
    try:
        stream = forecast_pairs(camera_frames(capture, args.size, args.fps, cv2), state,
                                lambda s, x: predict_next(params, args.architecture, s, x))
        for record in stream:
            actual = record["actual"]
            row = {"mse": float(np.mean((record["predicted"] - actual) ** 2)),
                   "persistence_mse": float(np.mean((record["last_frame"] - actual) ** 2)),
                   "frame_difference_extrapolation_mse": float(np.mean((record["extrapolated"] - actual) ** 2)),
                   "inference_ms": record["inference_ms"],
                   "prediction_committed_ns": record["prediction_committed_ns"],
                   "frame_arrived_ns": record["frame_arrived_ns"]}
            report["frames"].append(row)
            cv2.imshow("Causal Predictive Mirror (q to quit)", panel(record, args.scale, cv2))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {len(report['frames'])} frame comparisons to {output}")


if __name__ == "__main__":
    main()
