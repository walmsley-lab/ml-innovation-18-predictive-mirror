"""Synchronized side-by-side comparison of independent long-horizon predictors.

Live forecasts are committed before the next camera read. Delayed observations
score past predictions but never feed them backwards. Replay uses only the
final 20% of a local recording by default; future truth is displayed/scored
AFTER forecasts have been generated from past frames.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.horizon_phase import ARMS, rollout_one
from mirror import preprocess

BASELINE = "persistence"
METHODS = (BASELINE,) + ARMS


def load_models(folder):
    folder = Path(folder)
    models = {}
    for arm in ARMS:
        path = folder / (arm + ".npz")
        if not path.exists():
            raise FileNotFoundError(f"Missing independent checkpoint: {path}")
        with np.load(path, allow_pickle=False) as archive:
            models[arm] = {key: jnp.asarray(archive[key]) for key in archive.files}
        for key in ("local", "readout", "bias"):
            if key not in models[arm]:
                raise ValueError(f"{path} missing {key}")
    shapes = [{key: tuple(value.shape) for key, value in model.items()}
              for model in models.values()]
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("Comparison arms must have matching parameter shapes")
    return models


def predictors(models, horizon, coupling):
    """One separately compiled, frozen predictor per architecture."""
    return {arm: jax.jit(lambda observed, weights=weights, name=arm:
                         rollout_one(weights, observed, horizon, name, coupling))
            for arm, weights in models.items()}


def forecast_all(predict, context, horizon):
    """Identical observed context for every arm; never accepts future truth."""
    if len(context) < 2:
        raise ValueError("Need >= 2 observed frames")
    frames = jnp.asarray(context, jnp.float32)
    start = time.perf_counter()
    forecast = {arm: np.asarray(function(frames)) for arm, function in predict.items()}
    latency_ms = 1000.0 * (time.perf_counter() - start)
    for arm, array in forecast.items():
        if array.shape != (horizon, *context.shape[1:]):
            raise ValueError(f"Unexpected forecast shape from {arm}: {array.shape}")
    return forecast, latency_ms


def new_scores(leads):
    return {arm: {str(lead): {"sum_sq": 0.0, "pixels": 0,
                              "sum_motion_sq": 0.0, "motion_pixels": 0,
                              "samples": 0}
                  for lead in leads} for arm in METHODS}


def add_score(totals, lead, actual, anchor, predicted):
    """Future observation appears only here, in post-forecast scoring."""
    changed = np.abs(actual - anchor) > 0.03
    for arm, value in ((BASELINE, anchor), *predicted.items()):
        error = np.square(value[lead - 1] - actual) if arm != BASELINE else np.square(anchor - actual)
        slot = totals[arm][str(lead)]
        slot["sum_sq"] += float(error.sum())
        slot["pixels"] += int(error.size)
        slot["sum_motion_sq"] += float(error[changed].sum())
        slot["motion_pixels"] += int(changed.sum())
        slot["samples"] += 1


def summaries(totals):
    return {arm: {lead: {
        "mse": slot["sum_sq"] / slot["pixels"] if slot["pixels"] else None,
        "motion_region_mse": (slot["sum_motion_sq"] / slot["motion_pixels"]
                              if slot["motion_pixels"] else None),
        "motion_pixels": slot["motion_pixels"],
        "scored_forecasts": slot["samples"]}
        for lead, slot in by_lead.items()} for arm, by_lead in totals.items()}


def score_delayed(pending, frame_index, current, totals, leads):
    """Match live observations to forecasts issued at previous frame indices."""
    remaining = []
    for entry in pending:
        delta = frame_index - entry["index"]
        if delta in leads:
            add_score(totals, delta, current, entry["last"], entry["forecast"])
        if delta < max(leads):
            remaining.append(entry)
    return remaining


def validation_positions(count, context, horizon, stride=1, split="heldout"):
    """Context and all future truth remain within the chosen time region."""
    if stride < 1 or context < 2 or horizon < 1 or split not in ("heldout", "all"):
        raise ValueError("Invalid replay window configuration")
    start = int(0.8 * count) if split == "heldout" else context - 1
    start = max(start + context - 1 if split == "heldout" else start, context - 1)
    end = count - horizon
    if start > end:
        raise ValueError("Not enough frames in replay region")
    return range(start, end + 1, stride)


def draw_grid(current, forecast, leads, cv2, panel_size=144, actuals=None,
              metrics=None, latency_ms=None, fps=10):
    """All predictions share an anchor. Actual future frames are replay-only."""
    n_columns = 5
    gutter = 75
    header = 34
    width = gutter + n_columns * panel_size
    height = header + (1 + len(leads)) * panel_size
    canvas = np.full((height, width, 3), (23, 29, 38), np.uint8)
    column_names = ("Truth/replay", "Persistence", "Plain recurrent",
                    "Phase K=0", "Phase coupled")
    def write(message, x, y, color=(230, 240, 244), scale=0.38):
        cv2.putText(canvas, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
    def tile(array, col, row, label=None):
        if array is None:
            return
        pixel = np.asarray(array[..., 0])
        image = (255 * np.clip(pixel, 0, 1)).astype(np.uint8)
        rgb = cv2.cvtColor(cv2.resize(image, (panel_size, panel_size),
                                     interpolation=cv2.INTER_NEAREST),
                           cv2.COLOR_GRAY2BGR)
        x, y = gutter + col * panel_size, header + row * panel_size
        canvas[y:y+panel_size, x:x+panel_size] = rgb
        if label is not None:
            cv2.rectangle(canvas, (x, y+panel_size-23),
                          (x+panel_size, y+panel_size), (16, 23, 32), -1)
            write(label, x+5, y+panel_size-7, scale=0.36)
    for col, name in enumerate(column_names):
        write(name, gutter + col * panel_size + 4, 22)
    write("NOW", 10, header + panel_size//2)
    tile(current, 0, 0, "Current input")
    write("Same observed frame / shared timestamp", gutter+panel_size+10,
          header + panel_size//2)
    if latency_ms is not None:
        write(f"Total inference {latency_ms:.0f} ms", gutter+panel_size+10,
              header + panel_size//2+23, (120, 211, 227))
    for row, lead in enumerate(leads, 1):
        write(f"+{lead} fr", 4, header + row*panel_size + 18)
        write(f"~{lead/fps:.1f}s", 4, header + row*panel_size + 36,
              (140, 197, 206))
        tile(actuals.get(lead) if actuals else None, 0, row,
             "Actual future" if actuals else None)
        if actuals is None:
            write("Unavailable", gutter+7, header + row*panel_size+panel_size//2,
                  (130, 155, 165))
        tile(current, 1, row, "Last frame")
        for col, arm in enumerate(ARMS, 2):
            label = None
            if metrics is not None:
                val = metrics[arm][str(lead)]["mse"]
                label = f"MSE {val:.5f}" if val is not None else "Not scored"
            tile(forecast[arm][lead-1], col, row, label)
    return canvas


def run_replay(args, predict, cv2):
    with np.load(args.input, allow_pickle=False) as archive:
        frames = archive["frames"].astype(np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 1 or frames.shape[1] != args.size:
        raise ValueError("Recording frame size must match --size")
    positions = validation_positions(len(frames), args.context, args.horizon,
                                     args.stride, args.split)
    totals = new_scores(args.leads)
    runtimes = []
    frames_shown = 0
    for index in positions:
        context = frames[index-args.context+1:index+1]
        forecast, elapsed = forecast_all(predict, context, args.horizon)
        runtimes.append(elapsed)
        future = {lead: frames[index+lead] for lead in args.leads}
        for lead, true in future.items():
            add_score(totals, lead, true, frames[index], forecast)
        frames_shown += 1
        if not args.no_display:
            cv2.imshow("Predictive Mirror: synchronized replay (q to quit)",
                       draw_grid(frames[index], forecast, args.leads, cv2,
                                 args.panel_size, future, summaries(totals),
                                 elapsed, args.fps))
            if cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF in (ord("q"), 27):
                break
        if args.max_frames and frames_shown >= args.max_frames:
            break
    return frames_shown, totals, runtimes


def run_live(args, predict, cv2):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {args.camera}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    history = deque(maxlen=args.context)
    pending = []
    totals = new_scores(args.leads)
    runtimes = []
    frames_shown = 0
    index = -1
    deadline = time.monotonic()
    try:
        while True:
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            ok, raw = cap.read()
            if not ok:
                break
            index += 1
            current = preprocess(raw, args.size, cv2)
            history.append(current)
            pending = score_delayed(pending, index, current, totals, args.leads)
            deadline = max(deadline + 1.0 / args.fps, time.monotonic())
            if len(history) < args.context:
                continue
            forecast, elapsed = forecast_all(
                predict, np.stack(history), args.horizon)
            runtimes.append(elapsed)
            pending.append({"index": index, "last": current.copy(), "forecast": forecast})
            frames_shown += 1
            if not args.no_display:
                cv2.imshow("Predictive Mirror: synchronized LIVE (q to quit)",
                           draw_grid(current, forecast, args.leads, cv2,
                                     args.panel_size, None, summaries(totals),
                                     elapsed, args.fps))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and frames_shown >= args.max_frames:
                break
    finally:
        cap.release()
    return frames_shown, totals, runtimes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("live", "replay"), default="replay")
    p.add_argument("--parameters-dir", default="runs/horizon-phase")
    p.add_argument("--input", default="recordings/webcam.npz")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--coupling", type=float, default=0.25)
    p.add_argument("--panel-size", type=int, default=144)
    p.add_argument("--stride", type=int, default=1, help="Replay sampling step")
    p.add_argument("--split", choices=("heldout", "all"), default="heldout")
    p.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    p.add_argument("--out", default="runs/horizon-comparison.json")
    p.add_argument("--no-display", action="store_true")
    args = p.parse_args()
    args.leads = tuple(sorted(set(args.leads)))
    if (args.fps <= 0 or args.size < 8 or args.context < 2 or args.horizon < 1
            or args.panel_size < 80 or args.coupling <= 0
            or args.max_frames < 0 or not args.leads
            or args.leads[0] < 1 or args.leads[-1] > args.horizon):
        p.error("Invalid FPS, context, size, coupling, panel size, or lead times")
    if not args.no_display:
        try:
            import cv2
        except ImportError:
            p.error("Install opencv-python to display comparison")
    else:
        import cv2  # preprocess uses OpenCV for live capture; also supplied to replay
    models = load_models(args.parameters_dir)
    predict = predictors(models, args.horizon, args.coupling)
    if args.source == "live":
        print("LIVE: forecast each model from same frames; score after future capture. "
              "Device buffering and exposure times unverified.", flush=True)
        shown, totals, runtimes = run_live(args, predict, cv2)
    else:
        print("REPLAY: future truth is display/scoring-only; no future model inputs.",
              flush=True)
        shown, totals, runtimes = run_replay(args, predict, cv2)
    if not args.no_display:
        cv2.destroyAllWindows()
    report = {
        "source": args.source, "models": list(METHODS), "frames_displayed": shown,
        "leads_frames": list(args.leads), "nominal_fps": args.fps,
        "median_total_inference_ms": float(np.median(runtimes)) if runtimes else None,
        "metrics": summaries(totals),
        "provenance": "Replay split excludes train region by default; "
                      "live scores only after observing the future frame. "
                      "Single recording is not independent replication.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
