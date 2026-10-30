"""Deterministic 2D synthetic frames for testing developmental transfer."""

import numpy as np


def sequence(seed, steps=8, size=32, stage=0):
    """Visible grayscale frames only: [T+1,H,W,1], no privileged truth.

    Stages: moving dot, two moving dots, foreground occluder, camera shifts.
    """
    if not 0 <= stage <= 3 or size < 16 or steps < 1:
        raise ValueError("stage 0..3, size >=16, steps >=1 required")
    rng = np.random.default_rng(int(seed))
    objects = []
    for _ in range(1 if stage == 0 else 2):
        radius = int(rng.integers(1, max(3, size // 12 + 1)))
        x = float(rng.integers(radius + 2, size - radius - 2))
        y = float(rng.integers(radius + 2, size - radius - 2))
        vx = float(rng.choice((-2, -1, 1, 2)))
        vy = float(rng.choice((-1, 0, 1)))
        intensity = float(rng.uniform(0.68, 0.98))
        objects.append([x, y, vx, vy, radius, intensity])
    yy, xx = np.ogrid[:size, :size]
    camera = 0
    frames = []
    for t in range(steps + 1):
        frame = np.full((size, size), 0.08, np.float32)
        for x, y, _, _, radius, intensity in objects:
            disc = (xx - (x - camera)) ** 2 + (yy - y) ** 2 <= radius ** 2
            frame[disc] = intensity
        if stage >= 2:
            left = max(0, size // 2 - 2 - camera)
            right = min(size, size // 2 + 2 - camera)
            frame[:, left:right] = 0.32
        frames.append(frame[..., None])
        if t == steps:
            break
        for obj in objects:
            obj[0] += obj[2]
            obj[1] += obj[3]
            for coordinate, velocity in ((0, 2), (1, 3)):
                low, high = obj[4] + 1, size - obj[4] - 2
                if obj[coordinate] < low or obj[coordinate] > high:
                    obj[coordinate] = float(np.clip(obj[coordinate], low, high))
                    obj[velocity] = -obj[velocity]
        if stage == 3:
            camera = int(np.clip(camera + rng.choice((-1, 0, 1)), -2, 2))
    return np.stack(frames)


def batch(seeds, stages, steps=8, size=32):
    if len(seeds) != len(stages):
        raise ValueError("Each episode requires a stage")
    return np.stack([sequence(s, steps=steps, size=size, stage=int(stage))
                     for s, stage in zip(seeds, stages)])
