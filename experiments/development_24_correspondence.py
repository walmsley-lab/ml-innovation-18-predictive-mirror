"""Roadmap 24: causal visual correspondence across crossing and missing detections.

Two indistinguishable RGB blobs move with different trajectories. Detection is
from pixels; stable identities and invisible positions are evaluator-only.
This is a small classical identity floor, not learned object representations.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def world(seed=11, steps=9, size=32):
    if steps < 4 or size < 28:
        raise ValueError("Need at least four frames and size >= 28")
    rng = np.random.default_rng(seed)
    vertical = int(rng.integers(-1, 2))
    images = np.zeros((steps, size, size, 3), np.float32)
    truth = np.zeros((steps, 2, 2), np.float32)
    visible = np.ones((steps, 2), bool)
    for t in range(steps):
        truth[t, 0] = [3 + 3*t, 10 + vertical]
        truth[t, 1] = [size-5-3*t, 16 + vertical]
        visible[t, 0] = t not in (4, 5)
        for j, (x, y) in enumerate(truth[t].astype(int)):
            if visible[t, j] and 1 <= x < size-1:
                images[t, y-1:y+2, x-1:x+2] = .85
    return images, truth, visible


def detections(image, threshold=.5):
    """Find 8-connected bright blobs without object labels."""
    mask = np.asarray(image).mean(-1) > threshold
    seen = np.zeros_like(mask, bool)
    points = []
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        todo = [(int(y), int(x))]
        seen[y, x] = True
        group = []
        while todo:
            cy, cx = todo.pop()
            group.append((cx, cy))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy+dy, cx+dx
                    if (0 <= ny < height and 0 <= nx < width
                            and mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        todo.append((ny, nx))
        if len(group) >= 4:
            points.append(np.mean(group, axis=0))
    return np.asarray(points, dtype=float).reshape((-1, 2))


def match(predicted, found, max_distance=9.0):
    """One-to-one two-track assignment, allowing unmatched tracks."""
    best, cost = {}, float("inf")
    for indexes in itertools.permutations(range(len(predicted)),
                                          min(len(found), len(predicted))):
        proposal = {}
        value = 0.
        for detection, track in enumerate(indexes):
            dist = np.linalg.norm(predicted[track]-found[detection])
            if dist > max_distance:
                value += max_distance
                continue
            value += dist
            proposal[track] = found[detection]
        value += (len(predicted)-len(proposal))*max_distance
        if value < cost:
            best, cost = proposal, value
    return best


def track(frames, mode="coast"):
    if mode not in ("coast", "nearest"):
        raise ValueError("Unknown tracking control")
    first = detections(frames[0])
    if len(first) != 2:
        raise ValueError("Require two separate initial detections")
    position = first[np.argsort(first[:, 0])].copy()
    velocity = np.zeros_like(position)
    predicted = [position.copy()]
    for image in frames[1:]:
        guess = position + velocity if mode == "coast" else position
        found = detections(image)
        assigned = match(guess, found)
        next_position = guess.copy() if mode == "coast" else position.copy()
        for identity, point in assigned.items():
            observed_displacement = point-position[identity]
            velocity[identity] = .65*velocity[identity]+.35*observed_displacement
            next_position[identity] = point
        position = next_position
        predicted.append(position.copy())
    return np.stack(predicted)


def run(seed=11, out="runs/development-24/metrics.json"):
    frames, truth, visible = world(seed)
    rows = {}
    for mode in ("coast", "nearest"):
        prediction = track(frames, mode)
        error = np.linalg.norm(prediction-truth, axis=-1)
        rows[mode] = {
            "visible_mae_px": float(error[visible].mean()),
            "hidden_mae_px": float(error[~visible].mean()),
            "identity_crossing_mae_px": float(error[6:, :].mean()),
        }
    report = {"roadmap_id": 24, "seed": int(seed), "scores": rows,
              "note": "Same-appearance blobs with crossing and dropped detections. "
                      "Ground-truth identities and hidden locations are evaluator-only. "
                      "Track ordering originates from the first observed frame.",
              "limits": "Hand-designed pixel threshold/dynamics; not RGB-learned identity "
                        "or proof of object permanence."}
    if out is not None:
        p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/development-24/metrics.json")
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    print(json.dumps(run(a.seed, a.out), indent=2))


if __name__ == "__main__":
    main()
