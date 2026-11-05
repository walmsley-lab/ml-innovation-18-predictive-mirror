"""Roadmap 22: causal exposure adaptation with persistent and transient channels.

Pure NumPy diagnostic, not learned NCA plasticity or evidence about a retina.
Only observed RGB frames enter the encoder. Foreground truth is for scoring.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def channels(frames, decay=0.8):
    """Return normalized current appearance, persistent state, and change.

    Normalization is per observed frame; persistent state is a causal EMA.
    A pure temporal-difference encoder lacks the static appearance channel.
    """
    x = np.asarray(frames, dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] != 3 or len(x) < 2 or not 0 <= decay < 1:
        raise ValueError("Expected RGB [time,height,width,3] and 0<=decay<1")
    low = np.percentile(x, 5, axis=(1, 2), keepdims=True)
    high = np.percentile(x, 99.5, axis=(1, 2), keepdims=True)
    current = np.clip((x - low) / np.maximum(high - low, 1e-5), 0, 1)
    state = current[0].copy()
    persistent, change = [], []
    for image in current:
        change.append(image - state)
        state = decay * state + (1 - decay) * image
        persistent.append(state.copy())
    return current, np.stack(persistent), np.stack(change)


def world(seed=11, size=24, steps=16):
    if size < 16 or steps < 8:
        raise ValueError("World must accommodate a visible stationary object")
    rng = np.random.default_rng(seed)
    background = rng.uniform(.04, .12, (size, size, 3))
    mask = np.zeros((size, size), bool)
    mask[size//3:size//3+5, size//3:size//3+5] = True
    base = background.copy()
    base[mask] = [.78, .45, .25]
    exposure = np.resize(np.array([.48, .48, 1., 1., .65, .65, 1.25, 1.25]),
                         steps)
    frames = np.clip(exposure[:, None, None, None] * base[None], 0, 1)
    return frames.astype(np.float32), mask, exposure


def contrast(images, mask):
    fg = np.mean(images[:, mask], axis=(1, 2))
    bg = np.mean(images[:, ~mask], axis=(1, 2))
    return fg - bg


def run(seed=11, out="runs/development-22/metrics.json"):
    frames, mask, exposure = world(seed)
    normalized, persistent, change = channels(frames)
    raw = contrast(frames, mask)
    adapted = contrast(normalized, mask)
    retained = contrast(persistent, mask)
    result = {
        "roadmap_id": 22, "seed": int(seed), "exposures": exposure.tolist(),
        "raw_contrast_cv": float(np.std(raw) / np.maximum(np.mean(raw), 1e-9)),
        "normalized_contrast_cv": float(np.std(adapted) / np.maximum(np.mean(adapted), 1e-9)),
        "static_retained_contrast": float(retained[-1]),
        "change_only_static_contrast": float(contrast(np.abs(change), mask)[-1]),
        "controls": ["raw RGB", "pure temporal difference", "causal normalized appearance"],
        "limits": "Synthetic exposure invariance only; no trained visual adaptation, "
                  "future prediction, or webcam robustness established."
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="runs/development-22/metrics.json")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.out), indent=2))


if __name__ == "__main__":
    main()
