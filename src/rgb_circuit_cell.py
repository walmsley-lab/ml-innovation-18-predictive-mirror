"""Independent, CPU-only RGB adapter for the two-state circuit toy in NCA #2.

The identical local equation/coefficients are ported to NumPy for a fixed-rule
comparison. This is NOT an electrical device simulator or trained neural cell.
"""
import numpy as np

DEFAULT = dict(leak=0.18, coupling=0.12, drive_gain=0.35,
               memory_rate=0.04, nonlinear_gain=0.30)
MODES = ("full", "no_coupling", "no_feedback", "no_slow_memory", "quantized_4bit")


def _neighbors(v):
    # State shape [batch,height,width,color,2]; periodic four-neighbor grid.
    return (np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1)
            + np.roll(v, 1, axis=2) + np.roll(v, -1, axis=2)) * .25


def update(state, drive, mode="full"):
    """Simultaneous local step; exactly the portable NCA circuit equation."""
    if state.ndim != 5 or state.shape[-2:] != (3, 2) or drive.shape != state.shape[:-1]:
        raise ValueError("Expected state [B,H,W,3,2] and drive [B,H,W,3]")
    if mode not in MODES:
        raise ValueError("Unknown mode")
    p = DEFAULT.copy()
    if mode == "no_coupling":
        p["coupling"] = 0.
    elif mode == "no_feedback":
        p["nonlinear_gain"] = 0.
    elif mode == "no_slow_memory":
        p["memory_rate"] = 0.
    v, m = state[..., 0], state[..., 1]
    v_next = np.clip((1 - p["leak"]) * v
                     + p["coupling"] * (_neighbors(v) - v)
                     + p["drive_gain"] * drive
                     + p["nonlinear_gain"] * v * m, -1., 1.)
    m_next = np.clip(m + p["memory_rate"] * (np.tanh(2 * v_next) - m), -1., 1.)
    out = np.stack((v_next, m_next), axis=-1).astype(np.float32)
    if mode == "quantized_4bit":
        out = (np.round(out * 7) / 7).astype(np.float32)
    return out


def observe_context(context, mode="full", ticks=2):
    """Only observed context enters state; never read target frames."""
    context = np.asarray(context, np.float32)
    if context.ndim != 5 or context.shape[-1] != 3 or context.shape[1] < 2 or ticks < 1:
        raise ValueError("Expected RGB [B,C,H,W,3], C>=2, ticks>=1")
    b, _, h, w, _ = context.shape
    state = np.zeros((b, h, w, 3, 2), np.float32)
    for frame in np.moveaxis(context, 1, 0):
        for _ in range(ticks):
            state = update(state, frame * 2. - 1., mode)
    return state


def features(state, frame):
    """One shared four-number linear head; no nonlinear digital readout."""
    return np.stack((state[..., 0], state[..., 1], frame,
                     np.ones_like(frame)), axis=-1)


def fit_head(train, context=8, mode="full", ticks=2, ridge=1.):
    """Fit one shared ridge head on observed context -> NEXT-frame delta."""
    if train.shape[1] < context + 1 or ridge <= 0:
        raise ValueError("Need next-frame training labels and ridge>0")
    x = train[:, :context]
    s = observe_context(x, mode, ticks)
    design = features(s, x[:, -1]).reshape(-1, 4).astype(np.float64)
    target = (train[:, context] - x[:, -1]).reshape(-1).astype(np.float64)
    xtx = design.T @ design
    beta = np.linalg.solve(xtx + ridge * np.eye(4), design.T @ target)
    return beta.astype(np.float32)


def rollout(context, head, horizon=10, mode="full", ticks=2):
    """Autoregressive open-loop prediction: no future inputs or simulator truth."""
    if horizon < 1 or np.asarray(head).shape != (4,):
        raise ValueError("Positive horizon and four readout coefficients required")
    state = observe_context(context, mode, ticks)
    frame = np.asarray(context[:, -1], np.float32).copy()
    predictions = []
    for _ in range(horizon):
        residual = np.clip(features(state, frame) @ head, -.2, .2)
        frame = np.clip(frame + residual, 0., 1.).astype(np.float32)
        predictions.append(frame)
        for _ in range(ticks):
            state = update(state, frame * 2. - 1., mode)
    return np.stack(predictions, axis=1)
