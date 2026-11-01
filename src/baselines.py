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


def persistence_forecast(context, horizon):
    """Repeat the last observed RGB frame for ``horizon`` future frames."""
    context = np.asarray(context)
    if context.ndim != 4 or context.shape[-1] not in (1, 3) or horizon < 1:
        raise ValueError("Expected [T,H,W,C] context with C=1 or 3 and horizon >=1")
    return np.repeat(context[-1:,:,:,:], horizon, axis=0).copy()


def _warp_integer(frame, dy, dx):
    """Translate an image, filling newly exposed pixels with zero."""
    height, width = frame.shape[:2]
    output = np.zeros_like(frame)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    target_y0, target_y1 = max(0, dy), min(height, height + dy)
    target_x0, target_x1 = max(0, dx), min(width, width + dx)
    if source_y0 < source_y1 and source_x0 < source_x1:
        output[target_y0:target_y1, target_x0:target_x1] = \
            frame[source_y0:source_y1, source_x0:source_x1]
    return output


def _estimate_translation(previous, current, max_shift=6):
    """Estimate one global integer translation from two observed frames."""
    previous = np.mean(np.asarray(previous), axis=-1)
    current = np.mean(np.asarray(current), axis=-1)
    best = (0, 0)
    best_error = np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = _warp_integer(previous[..., None], dy, dx)[..., 0]
            error = float(np.mean(np.square(shifted - current)))
            if error < best_error:
                best_error, best = error, (dy, dx)
    return best


def constant_velocity_forecast(context, horizon, max_shift=6):
    """Warp the latest RGB frame using its observed global pixel velocity."""
    context = np.asarray(context)
    if context.ndim != 4 or context.shape[-1] not in (1, 3) or len(context) < 2:
        raise ValueError("Expected at least two [T,H,W,C] grayscale or RGB frames")
    if horizon < 1 or max_shift < 0:
        raise ValueError("horizon must be positive and max_shift nonnegative")
    dy, dx = _estimate_translation(context[-2], context[-1], max_shift)
    return np.stack([_warp_integer(context[-1], (step + 1) * dy,
                                  (step + 1) * dx)
                     for step in range(horizon)])


def optical_flow_forecast(context, horizon):
    """Dense Farneback optical-flow warp from the last two RGB observations."""
    context = np.asarray(context)
    if context.ndim != 4 or context.shape[-1] not in (1, 3) or len(context) < 2:
        raise ValueError("Expected at least two [T,H,W,C] grayscale or RGB frames")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("optical_flow_forecast requires opencv-python") from exc
    previous = np.mean(context[-2], axis=-1).astype(np.float32)
    current = np.mean(context[-1], axis=-1).astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(
        previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    height, width = current.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    forecasts = []
    for step in range(1, horizon + 1):
        # A source pixel at (x,y) moves to (x+u,y+v); sample the source at
        # the inverse map for each output pixel.
        map_x = grid_x - step * flow[..., 0]
        map_y = grid_y - step * flow[..., 1]
        warped = np.stack([cv2.remap(context[-1, ..., channel], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                           for channel in range(context.shape[-1])], axis=-1)
        forecasts.append(np.clip(warped, 0.0, 1.0))
    return np.stack(forecasts).astype(context.dtype, copy=False)
