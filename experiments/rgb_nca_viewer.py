"""Independent colorful RGB predictive-NCA evaluation and live mirror.

Replay displays future frames only after model forecasts are generated.
Live future frames are never available at prediction time; score past
forecasts only after the corresponding subsequent camera frame arrives.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.rgb_nca import preprocess_color, read_recording
from src import predictive_rgb_nca as nca


def infer(params, context, horizon, ticks):
    predicted, activity = nca.rollout_and_activity(
        params, jnp.asarray(context[None]), horizon, ticks)
    return np.asarray(predicted[0]), np.asarray(activity[0])


def score(prediction, last, truth):
    sq = np.mean(np.square(prediction - truth), axis=-1)
    base = np.mean(np.square(last - truth), axis=-1)
    changed = np.mean(np.abs(last - truth), axis=-1) > .03
    return {
        "model_mse": float(sq.mean()),
        "persistence_mse": float(base.mean()),
        "motion_region_model_mse": float(sq[changed].mean()) if changed.any() else None,
        "motion_region_persistence_mse": float(base[changed].mean()) if changed.any() else None,
        "motion_pixels": int(changed.sum()),
    }


def render_grid(context_last, forecasts, activity, leads, cv2, fps=10,
                actuals=None, panel_size=160):
    """Five-column display: future truth, RGB persistence, RGB forecast, error, NCA cells."""
    columns = 5
    heading, left = 45, 75
    canvas = np.full((heading + len(leads)*panel_size,
                      left + columns*panel_size, 3), (19, 29, 39), np.uint8)
    headers = ("Actual future", "Persistence", "RGB NCA forecast", "Abs RGB error", "NCA activity")
    for column, title in enumerate(headers):
        cv2.putText(canvas, title, (left + column*panel_size+4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (215, 240, 240), 1, cv2.LINE_AA)

    def tile(value, column, row, activity_map=False):
        if value is None:
            return
        if activity_map:
            gray = (255 * np.clip(value[..., 0] * 12.0, 0, 1)).astype(np.uint8)
            image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        else:
            rgb = (255 * np.clip(value, 0, 1)).astype(np.uint8)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (panel_size, panel_size),
                           interpolation=cv2.INTER_NEAREST)
        y, x = heading + row*panel_size, left + column*panel_size
        canvas[y:y+panel_size, x:x+panel_size] = image

    for row, lead in enumerate(leads):
        cv2.putText(canvas, f"+{lead}f", (7, heading+row*panel_size+18),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (180, 240, 215), 1)
        cv2.putText(canvas, f"~{lead/fps:.1f}s", (7, heading+row*panel_size+37),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (170, 170, 195), 1)
        truth = actuals.get(lead) if actuals else None
        tile(truth, 0, row)
        tile(context_last, 1, row)
        tile(forecasts[lead-1], 2, row)
        if truth is not None:
            tile(np.abs(forecasts[lead-1]-truth), 3, row)
        tile(activity[lead-1], 4, row, activity_map=True)
    return canvas


def replay(frames, predict, args, cv2):
    """Holdout context and targets stay strictly within final 20%."""
    start = int(.8 * len(frames)) + args.context - 1
    end = len(frames) - args.horizon
    if start > end:
        raise ValueError("Not enough frames in held-out replay")
    scores = {str(h): [] for h in args.leads}
    count, latencies = 0, []
    for i in range(start, end+1, args.stride):
        context = frames[i-args.context+1:i+1]
        before = time.perf_counter()
        prediction, cells = predict(jnp.asarray(context[None]))
        prediction, cells = np.asarray(prediction[0]), np.asarray(cells[0])
        latencies.append(1000*(time.perf_counter()-before))
        truths = {h: frames[i+h] for h in args.leads}
        for h, future in truths.items():
            scores[str(h)].append(score(prediction[h-1], context[-1], future))
        count += 1
        if not args.no_display:
            cv2.imshow("RGB NCA: held-out replay (q to quit)",
                       render_grid(context[-1], prediction, cells, args.leads, cv2,
                                   args.fps, truths, args.panel_size))
            if cv2.waitKey(max(1, int(1000/args.fps))) & 0xFF in (ord("q"), 27):
                break
        if args.max_frames and count >= args.max_frames:
            break
    return scores, latencies, count


def live(predict, args, cv2):
    camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"Color camera {args.camera} unavailable; check fuser /dev/video{args.camera}")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # advisory
    history = deque(maxlen=args.context)
    pending, scores, latencies = [], {str(h): [] for h in args.leads}, []
    deadline = time.monotonic()
    count, index = 0, -1
    try:
        while True:
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            valid, bgr = camera.read()
            if not valid:
                break
            index += 1
            current = preprocess_color(bgr, args.size, cv2)
            history.append(current)
            # Delayed scoring of earlier forecasts; never retroactively correct them.
            remaining = []
            for issued_at, anchor, forecast in pending:
                offset = index - issued_at
                if offset in args.leads:
                    scores[str(offset)].append(score(forecast[offset-1], anchor, current))
                if offset < max(args.leads):
                    remaining.append((issued_at, anchor, forecast))
            pending = remaining
            deadline = max(deadline + 1/args.fps, time.monotonic())
            if len(history) < args.context:
                continue
            before = time.perf_counter()
            prediction, cells = predict(jnp.asarray(np.stack(history)[None]))
            prediction, cells = np.asarray(prediction[0]), np.asarray(cells[0])
            latencies.append(1000*(time.perf_counter()-before))
            pending.append((index, current.copy(), prediction))
            count += 1
            if not args.no_display:
                cv2.imshow("RGB NCA: live forecast (q to quit)",
                           render_grid(current, prediction, cells, args.leads, cv2,
                                       args.fps, panel_size=args.panel_size))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and count >= args.max_frames:
                break
    finally:
        camera.release()
    return scores, latencies, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("replay", "live"), default="replay")
    parser.add_argument("--input", default="recordings/rgb-webcam.npz")
    parser.add_argument("--parameters", default="runs/rgb-nca/parameters.npz")
    parser.add_argument("--out", default="runs/rgb-nca/demo-metrics.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--context", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--leads", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--panel-size", type=int, default=160)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    args.leads = sorted(set(args.leads))
    if (args.horizon < 1 or args.context < 2 or args.ticks < 1 or args.fps <= 0
            or args.stride < 1 or args.max_frames < 0 or not args.leads
            or args.leads[0] < 1 or args.leads[-1] > args.horizon):
        parser.error("Invalid horizon/context/leads/ticks/FPS/stride")
    if args.source == "live" or not args.no_display:
        try:
            import cv2
        except ImportError:
            parser.error("Install opencv-python to show the demo or use the live camera")
    else:
        cv2 = None
    with np.load(args.parameters, allow_pickle=False) as archive:
        params = {k: jnp.asarray(archive[k]) for k in archive.files}
    predict = jax.jit(lambda x: nca.rollout_and_activity(params, x, args.horizon, args.ticks))
    if args.source == "live":
        rows, latencies, count = live(predict, args, cv2)
    else:
        frames, stamps = read_recording(args.input)
        if frames.shape[1] != args.size:
            raise ValueError("Recording frame size must match --size")
        rows, latencies, count = replay(frames, predict, args, cv2)
    if cv2 is not None and not args.no_display:
        cv2.destroyAllWindows()
    report = {"architecture": "predictive_rgb_nca", "source": args.source,
              "frames_forecast": count, "leads_frames": args.leads,
              "median_inference_ms": float(np.median(latencies)) if latencies else None,
              "score_by_horizon": {
                  horizon: {name: float(np.mean([row[name] for row in records if row[name] is not None]))
                            if any(row[name] is not None for row in records) else None
                            for name in ("model_mse", "persistence_mse",
                                         "motion_region_model_mse", "motion_region_persistence_mse")}
                  | {"scored_forecasts": len(records)}
                  for horizon, records in rows.items()},
              "caveat": "Live forecasts scored only as future observations arrive. "
                        "Camera buffer/exposure timing unverified; replay is a validation split, "
                        "not independent test data."}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
