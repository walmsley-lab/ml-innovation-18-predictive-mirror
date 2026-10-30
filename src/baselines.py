"""Observation-only next-frame baselines; simulator truth is never an input."""

import numpy as np


def persistence(frames):
    """Predict the last observed frame at each step [T+1,H,W,1]."""
    return np.asarray(frames[:-1]).copy()


def velocity(frames, actions):
    """Move bright observed objects by a two-frame velocity estimate."""
    frames = np.asarray(frames)
    actions = np.asarray(actions)
    result = []
    for t in range(len(actions)):
        current = frames[t].copy()
        predicted = np.zeros_like(current)
        pan = int(actions[t, 0])
        # Shift existing scene by known camera motion; no hidden object or mask used.
        if pan < 0:
            predicted[-pan:] = current[:pan]
        elif pan > 0:
            predicted[:-pan] = current[pan:]
        else:
            predicted[:] = current
        pixel = np.argwhere(current[..., 0] > 0.9)
        prev = np.argwhere(frames[t - 1, ..., 0] > 0.9) if t else np.empty((0, 2))
        if len(pixel) == 1 and len(prev) == 1:
            vy = int(pixel[0, 0] - prev[0, 0])
            vx = int(pixel[0, 1] - prev[0, 1]) + int(actions[t - 1, 0])
        else:
            vy = vx = 0
        predicted[..., 0][predicted[..., 0] > 0.9] = 0.0
        if len(pixel) == 1:
            y, x = (int(pixel[0, 0] + vy), int(pixel[0, 1] + vx - pan))
            if 0 <= y < current.shape[0] and 0 <= x < current.shape[1]:
                predicted[y, x, 0] = 1.0
        result.append(predicted)
    return np.stack(result)
