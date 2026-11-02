"""Visual, observation-only RGB NCA gallery. Pillow renders optional PNGs."""
import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark


def predict_from_context(predict, frames, context):
    """No future images reach the model; only the observed prefix."""
    if context < 2 or frames.ndim != 5 or frames.shape[1] <= context:
        raise ValueError("Expected RGB episodes with context and future frames")
    return predict(frames[:, :context])


def render(frames, forecast, activity, baselines, context, leads, output, scale=5):
    """One row per lead; truth is for visual evaluation only."""
    from PIL import Image, ImageDraw

    def tile(image):
        pixels = (255 * np.clip(image, 0, 1)).astype(np.uint8)
        return Image.fromarray(pixels, "RGB").resize(
            (pixels.shape[1] * scale, pixels.shape[0] * scale),
            Image.Resampling.NEAREST)

    labels = ("Observed", "RGB NCA", "Persistence", "Velocity",
              "Future truth", "Abs error", "NCA activity")
    height, width = frames.shape[1:3]
    tw, th = width * scale, height * scale
    canvas = Image.new("RGB", (56 + tw*len(labels),
                               38 + len(leads)*(th+22)), (19, 29, 39))
    draw = ImageDraw.Draw(canvas)
    for j, name in enumerate(labels):
        draw.text((56 + j*tw+3, 12), name, fill=(222, 240, 245))
    for row, lead in enumerate(leads):
        actual = frames[context+lead-1]
        forecast_rgb = forecast[lead-1]
        signal = np.clip(activity[lead-1, ..., 0] * 12, 0, 1)
        colored_activity = np.stack((signal, .3*signal, 1-signal), axis=-1)
        images = (frames[context-1], forecast_rgb,
                  baselines["persistence"][lead-1],
                  baselines["constant_velocity"][lead-1], actual,
                  np.abs(forecast_rgb-actual), colored_activity)
        y = 38 + row*(th+22)
        draw.text((6, y+4), f"+{lead}f", fill=(210, 231, 236))
        for col, image in enumerate(images):
            canvas.paste(tile(image), (56+col*tw, y))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def run(parameters, output="runs/rgb-prediction-gallery", seed=71000, episodes=3,
        context=8, horizon=10, size=32, ticks=2, scale=5):
    if episodes < 1 or context < 2 or horizon < 1 or ticks < 1 or scale < 1:
        raise ValueError("Invalid experiment settings")
    checkpoint = Path(parameters)
    with np.load(checkpoint, allow_pickle=False) as archive:
        weights = {name: jnp.asarray(archive[name]) for name in archive.files}
    x = benchmark.frames(benchmark.scenes(seed, episodes, context, horizon, size))
    model, activity = predict_from_context(
        lambda observed: nca.rollout_and_activity(
            weights, jnp.asarray(observed), horizon, ticks), x, context)
    model, activity = np.asarray(model), np.asarray(activity)
    controls = benchmark.controls(x[:, :context], horizon)
    leads = tuple(sorted(set(h for h in (*benchmark.LEADS, horizon) if h <= horizon)))
    metrics = {name: benchmark.score(forecast, x, context, leads)
               for name, forecast in {"nca": model, **controls}.items()}
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [render(x[i], model[i], activity[i],
                    {name: value[i] for name, value in controls.items()},
                    context, leads, folder/f"episode-{i:02d}.png", scale)
             for i in range(episodes)]
    report = {"experiment": "rgb_nca_frozen_prediction_gallery",
              "checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "test_seed_start": seed, "episodes": episodes, "context": context,
              "horizon": horizon, "leads": list(leads), "ticks": ticks,
              "metrics": metrics, "images": paths,
              "caveat": "Scenes are held out only if excluded from training/selection. "
                        "Only past frames reach the model and controls. "
                        "Cell activity is diagnostic, not decoded object identity."}
    (folder/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters", required=True)
    p.add_argument("--out", default="runs/rgb-prediction-gallery")
    p.add_argument("--seed", type=int, default=71000)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--scale", type=int, default=5)
    a = p.parse_args()
    result = run(a.parameters, a.out, a.seed, a.episodes,
                 a.context, a.horizon, a.size, a.ticks, a.scale)
    print(json.dumps({"images": result["images"],
                      "metrics": str(Path(a.out)/"metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
