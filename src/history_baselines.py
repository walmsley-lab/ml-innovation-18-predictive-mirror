"""Observation-only one-object history baselines for controlled RGB occlusion.

No palette IDs, simulator positions, velocities, occluder geometry, or future
frames. Assumes a static background and one visually saturated moving blob;
this assumption is declared and tested, not hidden as learned perception.
"""
import numpy as np


def observations(context):
    """Detect high-chroma moving pixels using only the observed RGB prefix."""
    rgb = np.asarray(context, np.float64)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or len(rgb) < 3:
        raise ValueError("Expected RGB context [time,height,width,3], >=3 frames")
    found = []
    for t, frame in enumerate(rgb):
        high = frame.max(axis=-1)
        low = frame.min(axis=-1)
        mask = (high - low > 0.15) & (high > 0.30)
        if mask.sum() < 5:
            continue
        yy, xx = np.nonzero(mask)
        found.append((t, float(xx.mean()), float(yy.mean()),
                      int(mask.sum()), np.median(frame[mask], axis=0)))
    return found


def estimate(context):
    """Estimate position, velocity, radius and color from visible observations."""
    points = observations(context)
    if len(points) < 2:
        return None
    # Discard partial detections: use the largest ~full-disc observations.
    max_area = max(p[3] for p in points)
    stable = [p for p in points if p[3] >= 0.75 * max_area]
    if len(stable) < 2:
        stable = points
    times = np.asarray([p[0] for p in stable], np.float64)
    xy = np.asarray([[p[1], p[2]] for p in stable], np.float64)
    if len(np.unique(times)) < 2:
        return None
    velocities = np.array([np.polyfit(times, xy[:, axis], 1)[0]
                           for axis in range(2)])
    intercept = xy.mean(axis=0) - velocities * times.mean()
    area = np.median([p[3] for p in stable])
    radius = max(1., float(np.sqrt(area / np.pi)))
    color = np.median(np.stack([p[4] for p in stable]), axis=0)
    return {"intercept": intercept, "velocity": velocities,
            "radius": radius, "color": color, "measurements": points}


def _background_and_occluder(context):
    # Use only the last observed frame, whose object may be behind the barrier
    # in the paired-history task. This is a visual estimate, not oracle mask.
    frame = np.asarray(context[-1], np.float64)
    median_rgb = np.median(frame.reshape(-1, 3), axis=0)
    neutral = frame.max(axis=-1) - frame.min(axis=-1) < 0.10
    static_structure = neutral & (np.max(np.abs(frame - median_rgb), axis=-1) > .04)
    return frame.copy(), static_structure


def _render(context, positions, radius, color):
    background, structure = _background_and_occluder(context)
    rows, cols = np.indices(background.shape[:2])
    result = []
    for x, y in positions:
        frame = background.copy()
        disc = (cols - x) ** 2 + (rows - y) ** 2 <= radius ** 2
        frame[disc & ~structure] = color
        result.append(np.clip(frame, 0, 1))
    return np.asarray(result, np.float32)


def history_linear(context, horizon=10):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    context = np.asarray(context)
    model = estimate(context)
    if model is None:
        return np.repeat(context[-1:], horizon, axis=0).astype(np.float32)
    times = np.arange(len(context), len(context) + horizon)
    positions = [model["intercept"] + model["velocity"] * t for t in times]
    return _render(context, positions, model["radius"], model["color"])


def kalman_history(context, horizon=10, process_noise=.05, measurement_noise=.5):
    """Constant-velocity Kalman estimator updated by observed RGB detections."""
    if horizon < 1 or process_noise <= 0 or measurement_noise <= 0:
        raise ValueError("Invalid Kalman configuration")
    context = np.asarray(context)
    model = estimate(context)
    if model is None:
        return np.repeat(context[-1:], horizon, axis=0).astype(np.float32)
    points = model["measurements"]
    by_time = {p[0]: np.array([p[1], p[2]], float) for p in points}
    first = points[0][0]
    # State is [x, y, vx, vy]. Fixed dynamics / measurement noise are declared
    # assumptions; measurements are extracted from RGB, not privileged labels.
    state = np.r_[by_time[first], model["velocity"]]
    cov = np.eye(4) * 2.
    trans = np.array([[1., 0., 1., 0.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
    H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    Q = np.eye(4) * process_noise
    R = np.eye(2) * measurement_noise
    for t in range(first + 1, len(context)):
        state = trans @ state
        cov = trans @ cov @ trans.T + Q
        if t in by_time:
            residual = by_time[t] - H @ state
            innovation = H @ cov @ H.T + R
            gain = np.linalg.solve(innovation, H @ cov).T
            state = state + gain @ residual
            cov = (np.eye(4) - gain @ H) @ cov
    positions = []
    for _ in range(horizon):
        state = trans @ state
        positions.append(state[:2].copy())
    return _render(context, positions, model["radius"], model["color"])
