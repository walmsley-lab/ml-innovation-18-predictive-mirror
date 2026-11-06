"""Train-only linear controls and diagnostics for frozen latent representations."""

import numpy as np


def fit(features, targets, penalty=1.):
    x, y = np.asarray(features, float), np.asarray(targets, float)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2
            or penalty <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Need finite matched feature/target matrices and positive ridge")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-4)
    center = y.mean(axis=0)
    x = (x - mean) / scale
    # Dual solve is small when feature dimension exceeds the episode count.
    weights = x.T @ np.linalg.solve(x @ x.T + penalty * np.eye(len(x)), y - center)
    return {"mean": mean, "scale": scale, "weights": weights, "bias": center}


def predict(params, features):
    return (np.asarray(features) - params["mean"]) / params["scale"] @ params["weights"] + params["bias"]


def score(predicted, actual):
    error = np.mean((np.asarray(predicted) - actual)**2, axis=0)
    variance = np.var(actual, axis=0)
    return {"mse": float(error.mean()), "mse_by_coordinate": error.tolist(),
            "r2_by_coordinate": [float(1 - e/v) if v > 1e-12 else None
                                 for e, v in zip(error, variance)]}


def pooled(clips, grid=4):
    b, t, h, w, c = clips.shape
    if h % grid or w % grid:
        raise ValueError("Clip geometry must be divisible by pooling grid")
    return clips.reshape(b, t, grid, h//grid, grid, w//grid, c).mean((3, 5))


def image_features(context, last_only=False):
    x = pooled(context[:, -1:] if last_only else context)
    return x.reshape(len(x), -1)


def fit_history_forecast(clips, context, horizon):
    prefix = clips[:, :context]
    residual = pooled(clips[:, context:context+horizon] - clips[:, context-1:context])
    return fit(image_features(prefix), residual.reshape(len(clips), -1))


def history_forecast(params, context, horizon):
    # Nearest expansion is deliberately a simple, declared coarse control.
    b, _, h, w, _ = context.shape
    residual = predict(params, image_features(context)).reshape(b, horizon, 4, 4, 3)
    residual = residual.repeat(h//4, axis=2).repeat(w//4, axis=3)
    return np.clip(context[:, -1:] + residual, 0, 1).astype(np.float32)
