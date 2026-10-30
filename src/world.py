"""Tiny deterministic camera world. Hidden truth is never a training target."""

import numpy as np


def sequence(seed, steps=12, size=16, occlusion=True, camera=True):
    """Return observable frames [T+1,H,W,1], camera deltas [T,1], hidden truth.

    A dot moves in world coordinates, sometimes behind a fixed vertical wall.
    The sensor can translate horizontally; only its movement command is observed.
    """
    rng = np.random.default_rng(seed)
    width = size + 6
    x = int(rng.integers(3, width - 3))
    direction = int(rng.choice([-1, 1]))
    y = int(rng.integers(3, size - 3))
    view = 3
    wall_x = width // 2
    positions, views, frames, actions = [], [], [], []
    for t in range(steps + 1):
        frame = np.zeros((size, size, 1), np.float32)
        if occlusion:
            wx = wall_x - view
            if 0 <= wx < size:
                frame[:, wx, 0] = 0.35
        px = x - view
        behind_wall = occlusion and x == wall_x
        if 0 <= px < size and not behind_wall:
            frame[y, px, 0] = 1.0
        frames.append(frame)
        positions.append(x)
        views.append(view)
        if t < steps:
            delta = int(rng.choice([-1, 0, 1])) if camera else 0
            delta = int(np.clip(view + delta, 0, width - size) - view)
            actions.append([delta])
            view += delta
            x += direction
            if x < 1 or x >= width - 1:
                direction = -direction
                x += 2 * direction
    return (np.stack(frames), np.asarray(actions, np.float32),
            {"world_x": np.asarray(positions), "camera_x": np.asarray(views)})


def batch(seeds, steps=12, size=16, occlusion=True, camera=True):
    runs = [sequence(int(s), steps, size, occlusion, camera) for s in seeds]
    return (np.stack([r[0] for r in runs]),
            np.stack([r[1] for r in runs]))
