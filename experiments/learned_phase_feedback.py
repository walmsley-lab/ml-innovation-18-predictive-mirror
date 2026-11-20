"""Learned phase-feedback readout on held-out occlusion sequences.

CPU-only NumPy baseline. The recurrent latent is a noisy persistence estimate;
a learned ridge readout combines that state with feedback, with or without
phase-dependent interactions. This does NOT learn an oscillator or phase itself.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def sequences(seed, episodes=120, length=32, phase_informative=True):
    rng = np.random.default_rng(seed)
    x = np.zeros((episodes, length))
    x[:, 0] = rng.normal(size=episodes)
    for t in range(1, length):
        x[:, t] = 0.93 * x[:, t - 1] + rng.normal(0, 0.35, episodes)
    phase = rng.uniform(-np.pi, np.pi, (episodes, 1)) + np.arange(length)[None, :] * 0.65
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    visible = rng.uniform(size=x.shape) > 0.40
    sensory = np.where(visible, x + rng.normal(0, 0.45, x.shape), 0.0)
    reliability = (0.5 + 0.5 * np.cos(phase)) if phase_informative else np.full_like(x, 0.5)
    feedback = x + rng.normal(size=x.shape) * (0.15 + 1.8 * (1 - reliability))
    # State only uses current and past sensory samples, never future truth.
    state = np.zeros_like(x)
    for t in range(length):
        state[:, t] = (0.8 * state[:, t - 1] if t else 0) + 0.2 * sensory[:, t]
    return dict(x=x, phase=phase, visible=visible, sensory=sensory,
                feedback=feedback, state=state)


def features(data, mode, phase_override=None):
    s = data["state"].reshape(-1)
    f = data["feedback"].reshape(-1)
    v = data["visible"].astype(float).reshape(-1)
    p = (data["phase"] if phase_override is None else phase_override).reshape(-1)
    common = [np.ones_like(s), s, f, v, data["sensory"].reshape(-1)]
    if mode == "phase":
        extras = [f * np.cos(p), f * np.sin(p)]
    elif mode == "nonphase":
        # Same number of fitted coefficients, using phase-blind nonlinear cues.
        extras = [f * np.abs(s), f * s]
    elif mode == "plain":
        extras = [np.zeros_like(f), np.zeros_like(f)]
    else:
        raise ValueError(mode)
    return np.stack(common + extras, axis=-1)


def fit(train, mode, alpha):
    a = features(train, mode)
    y = train["x"].reshape(-1)
    penalty = np.eye(a.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(a.T @ a + penalty, a.T @ y)


def mse(data, mode, weights, phase_override=None):
    prediction = features(data, mode, phase_override) @ weights
    target = data["x"].reshape(-1)
    missing = ~data["visible"].reshape(-1)
    return float(np.mean((prediction[missing] - target[missing]) ** 2))


def run_seed(seed, episodes=120, length=32, phase_informative=True):
    train = sequences(seed + 1000, episodes, length, phase_informative)
    valid = sequences(seed + 2000, episodes // 2, length, phase_informative)
    test = sequences(seed + 3000, episodes // 2, length, phase_informative)
    result = {}
    for mode in ("phase", "nonphase", "plain"):
        candidates = [(mse(valid, mode, fit(train, mode, a)), a)
                      for a in (0.01, 0.1, 1.0, 10.0)]
        alpha = min(candidates)[1]
        w = fit(train, mode, alpha)
        result[mode] = mse(test, mode, w)
        if mode == "phase":
            shuffled = np.random.default_rng(seed + 4000).permutation(
                test["phase"].reshape(-1)).reshape(test["phase"].shape)
            result["phase_shuffled_test"] = mse(test, mode, w, shuffled)
    result["state_only"] = float(np.mean(
        (test["state"][~test["visible"]] - test["x"][~test["visible"]]) ** 2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 41])
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("runs/learned-phase-feedback.json"))
    args = p.parse_args(argv)
    if args.episodes < 20 or args.length < 4:
        p.error("Require at least 20 episodes and 4 time steps")
    rows = [{"seed": seed, "phase_informative": informative,
             **run_seed(seed, args.episodes, args.length, informative)}
            for informative in (True, False) for seed in args.seeds]
    result = {"protocol": "learned_phase_feedback_v1", "rows": rows,
              "scope": "Ridge-trained readout of fixed recurrent sensory state; phase is supplied."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
