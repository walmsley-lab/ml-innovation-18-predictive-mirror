"""Side-by-side forecast galleries that show *why* the event score came out.

For each episode and lead this renders Observed | Future truth | each model
forecast, and overlays, per visible object:

* a ring at the true soft centroid,
* a dot at the predicted soft centroid (omitted when the forecast has no mass),
* the predicted colour-mass fraction and whether the hard threshold detector
  found the object.

It always writes a JSON manifest with those numbers (so it is useful headless
and testable without Pillow) and, when Pillow is installed, the PNG panels.
Future frames are used for visualization and scoring only; the model sees only
the observed context.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src import rgb_benchmark, rgb_event_metrics as em, rgb_motion_world
from src.rgb_motion_world import COLORS


def _found(image, color):
    return em._center(image, color) is not None


def manifest(forecasts_by_model, records, context, leads):
    """Per-episode, per-lead, per-object soft/hard localization summary."""
    episodes = []
    for i, record in enumerate(records):
        frames, visible = record["frames"], record["visible"]
        leads_out = {}
        for lead in leads:
            target_index = context + lead - 1
            target = frames[target_index]
            objects = []
            for obj, is_visible in enumerate(visible[target_index]):
                if not is_visible or em._center(target, COLORS[obj]) is None:
                    continue
                true_mass, true_center = em.soft_localization(target, COLORS[obj])
                entry = {"object": obj,
                         "true_soft_centroid": true_center.tolist(),
                         "models": {}}
                for name, forecast in forecasts_by_model.items():
                    prediction = forecast[i, lead - 1]
                    mass, center = em.soft_localization(prediction, COLORS[obj])
                    entry["models"][name] = {
                        "hard_found": bool(_found(prediction, COLORS[obj])),
                        "mass_fraction": (mass / true_mass) if true_mass > 1e-9 else 0.0,
                        "soft_centroid": None if center is None else center.tolist(),
                    }
                objects.append(entry)
            leads_out[str(lead)] = objects
        episodes.append({"episode": i, "seed_offset": i, "leads": leads_out})
    return episodes


def render(forecasts_by_model, records, context, leads, folder, scale=7):
    """Write one PNG panel per episode when Pillow is available; else skip."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []

    def tile(image):
        pixels = (255 * np.clip(image, 0, 1)).astype(np.uint8)
        return Image.fromarray(pixels, "RGB").resize(
            (pixels.shape[1] * scale, pixels.shape[0] * scale),
            Image.Resampling.NEAREST)

    def mark(draw, x0, y0, center, outline, fill=None, r=3):
        cy, cx = center
        px, py = x0 + cx * scale, y0 + cy * scale
        draw.ellipse([px - r, py - r, px + r, py + r], outline=outline,
                     fill=fill, width=2)

    names = list(forecasts_by_model)
    columns = ["Observed", "Future truth", *[n.upper() for n in names]]
    height, width = records[0]["frames"].shape[1:3]
    tw, th = width * scale, height * scale
    paths = []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        frames, visible = record["frames"], record["visible"]
        canvas = Image.new("RGB", (60 + tw * len(columns),
                                   40 + len(leads) * (th + 26)), (18, 27, 37))
        draw = ImageDraw.Draw(canvas)
        for j, name in enumerate(columns):
            draw.text((60 + j * tw + 3, 12), name, fill=(223, 240, 245))
        for row, lead in enumerate(leads):
            target_index = context + lead - 1
            target = frames[target_index]
            images = [frames[context - 1], target,
                      *[forecasts_by_model[n][i, lead - 1] for n in names]]
            y = 40 + row * (th + 26)
            draw.text((6, y + 4), f"+{lead}f", fill=(210, 231, 236))
            for col, image in enumerate(images):
                x0 = 60 + col * tw
                canvas.paste(tile(image), (x0, y))
                if col == 0:
                    continue  # observed frame: no overlay
                for obj, is_visible in enumerate(visible[target_index]):
                    if not is_visible or em._center(target, COLORS[obj]) is None:
                        continue
                    _, true_center = em.soft_localization(target, COLORS[obj])
                    mark(draw, x0, y, true_center, outline=(90, 240, 120))
                    _, pred_center = em.soft_localization(image, COLORS[obj])
                    if pred_center is not None:
                        mark(draw, x0, y, pred_center, outline=(250, 90, 90),
                             fill=(250, 90, 90), r=2)
        path = folder / f"episode-{i:02d}.png"
        canvas.save(path)
        paths.append(str(path))
    return paths


def run(out="runs/rgb-event-gallery", checkpoint=None, seed=81000, episodes=4,
        context=8, horizon=10, size=32, balls=2, occlusion=True,
        camera_motion=False, ticks=2, scale=7):
    if episodes < 1 or context < 2 or horizon < 1 or balls < 1 or balls > 4:
        raise ValueError("Invalid gallery dimensions")
    records = [rgb_motion_world.episode(
        seed + i, steps=context + horizon, size=size, balls=balls,
        occlusion=occlusion, camera_motion=camera_motion) for i in range(episodes)]
    frames = rgb_benchmark.frames(records)
    predicts = rgb_benchmark.controls(frames[:, :context], horizon)
    if checkpoint is not None:
        import jax.numpy as jnp
        from src import predictive_rgb_nca as nca
        with np.load(checkpoint, allow_pickle=False) as archive:
            params = {k: jnp.asarray(archive[k]) for k in archive.files}
        predicts = {"nca": np.asarray(nca.rollout(
            params, jnp.asarray(frames[:, :context]), horizon, ticks)), **predicts}
    leads = tuple(sorted({horizon, *(x for x in (1, 3, 5, 10) if x <= horizon)}))
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenario": {"seed_start": seed, "episodes": episodes, "balls": balls,
                     "occlusion": occlusion, "camera_motion": camera_motion,
                     "context": context, "horizon": horizon, "size": size},
        "checkpoint": str(checkpoint) if checkpoint else None,
        "models": list(predicts),
        "episodes": manifest(predicts, records, context, leads),
    }
    (folder / "gallery.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["panels"] = render(predicts, records, context, leads, folder, scale)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--out", default="runs/rgb-event-gallery")
    p.add_argument("--seed", type=int, default=81000)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--balls", type=int, default=2)
    p.add_argument("--no-occlusion", action="store_true")
    p.add_argument("--camera-motion", action="store_true")
    p.add_argument("--scale", type=int, default=7)
    a = p.parse_args()
    summary = run(a.out, a.checkpoint, a.seed, a.episodes, a.context, a.horizon,
                  a.size, a.balls, not a.no_occlusion, a.camera_motion, scale=a.scale)
    panels = summary.get("panels") or []
    print(f"Wrote {Path(a.out)/'gallery.json'} and {len(panels)} PNG panel(s).")


if __name__ == "__main__":
    main()
