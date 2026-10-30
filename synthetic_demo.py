"""Render an honest synthetic occlusion demo from a frozen checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from evaluate import score
from src import model
from src.baselines import persistence
from src.world import sequence


DEFAULT_SEEDS = {1: 60_000_000, 3: 60_000_001, 5: 60_000_002}


def load_parameters(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: jnp.asarray(archive[key]) for key in archive.files}


def event_metrics(forecast, frames, truth):
    """Measure reappearance events, including explicit detection counts."""
    visible = truth["visible_mask"]
    occluded = truth["occluded"]
    errors = []
    detected = 0
    had_object = bool(visible[0].any())
    for t in range(1, len(visible)):
        if had_object and bool(occluded[t - 1]) and visible[t].any():
            target = np.argwhere(visible[t, ..., 0])[0]
            estimate = forecast[t - 1, ..., 0]
            if float(estimate.max()) >= 0.5:
                detected += 1
                predicted = np.array(np.unravel_index(np.argmax(estimate), estimate.shape))
                errors.append(float(np.linalg.norm(predicted - target, ord=1)))
            else:
                errors.append(float(frames.shape[1]))
        had_object |= bool(visible[t].any())
    count = len(errors)
    return {
        "event_count": count,
        "detected_event_count": detected,
        "detection_rate": detected / count if count else None,
        "localization_mae_px": sum(errors) / count if count else None,
        "occluded_frame_count": int(np.count_nonzero(occluded)),
    }


def collect(params, widths=(1, 3, 5), seeds=None, steps=20):
    seeds = dict(DEFAULT_SEEDS if seeds is None else seeds)
    results = {}
    episodes = {}
    for width in widths:
        seed = int(seeds[width])
        frames, actions, truth = sequence(seed, steps=steps, wall_width=width)
        forecast = np.asarray(model.predict(params, frames, actions))
        baseline = persistence(frames)
        results[str(width)] = {
            "seed": seed,
            "model": {**score(forecast, frames, truth["visible_mask"], truth["occluded"]),
                       **event_metrics(forecast, frames, truth)},
            "persistence": {**score(baseline, frames, truth["visible_mask"], truth["occluded"]),
                             **event_metrics(baseline, frames, truth)},
        }
        episodes[str(width)] = {
            "frames": frames,
            "forecast": forecast,
            "baseline": baseline,
            "truth": truth,
        }
    return results, episodes


def render(episodes, output):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Rendering requires optional dependency: pip install Pillow") from exc

    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    scale, panel = 12, 16 * 12
    labels = ("target", "recurrent", "persistence", "recurrent error")
    for width, episode in episodes.items():
        frames = episode["frames"][1:]
        forecasts = episode["forecast"]
        baseline = episode["baseline"]
        truth = episode["truth"]
        rendered = []
        for index, (target, prediction, previous) in enumerate(zip(frames, forecasts, baseline)):
            error = np.abs(prediction - target)
            canvas = Image.new("RGB", (panel * 4, panel + 34), "white")
            draw = ImageDraw.Draw(canvas)
            occluded = bool(truth["occluded"][index + 1])
            marker = "OCCLUDED" if occluded else "visible"
            draw.text((4, 3), f"wall={width} frame={index + 1} {marker}", fill="black", font=font)
            for column, (label, values) in enumerate(zip(labels, (target, prediction, previous, error))):
                values = np.clip(values[..., 0] * 255.0, 0, 255).astype(np.uint8)
                image = Image.fromarray(values).resize((panel, panel), Image.Resampling.NEAREST).convert("RGB")
                canvas.paste(image, (column * panel, 22))
                draw.text((column * panel + 4, 22 + 4), label, fill="white", font=font)
            rendered.append(canvas)
        rendered[0].save(output / f"occlusion-{width}.gif", save_all=True,
                         append_images=rendered[1:], duration=180, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--output", default="runs/demo-v0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("steps must be at least 2")
    parameters = Path(args.parameters)
    output = Path(args.output)
    params = load_parameters(parameters)
    metrics, episodes = collect(params, steps=args.steps)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "demo": "synthetic-occlusion-v0",
        "architecture": "recurrent",
        "parameter_sha256": hashlib.sha256(parameters.read_bytes()).hexdigest(),
        "evaluation_seeds": DEFAULT_SEEDS,
        "steps": args.steps,
        "ground_truth_used_for": "event labels and visualization annotations only",
        "metrics": metrics,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if not args.no_render:
        render(episodes, output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
