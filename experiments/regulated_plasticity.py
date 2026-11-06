"""Does surprise-regulated plasticity improve adaptation without erasing memory?

A deterministic, continuous two-object RGB world changes its motion law and
then returns to the original one. All arms see identical observations.
Forecasts are scored BEFORE the new frame is absorbed. This is an online
plasticity probe, not evidence of consciousness or a chemical mechanism.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import DevelopmentalSheet, _pool


MODES = ("stable", "explore", "consolidate")


def world(seed=11, size=32, segment_frames=40):
    """One continuous scene: familiar motion -> reversal -> familiar motion."""
    if size < 16 or segment_frames < 8:
        raise ValueError("Need size >= 16 and at least eight frames per regime")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    position = np.array([size * .40, size * .55], np.float64)
    velocity = np.array([.65, .17], np.float64)
    anchor = rng.uniform(size * .25, size * .75, 2)
    pictures = []
    for t in range(3 * segment_frames):
        if t == segment_frames:
            velocity = np.array([-1.0, -.25])
        elif t == 2 * segment_frames:
            velocity = np.array([.65, .17])
        moving = np.exp(-((x - position[0]) ** 2 + (y - position[1]) ** 2) / 8.)
        landmark = np.exp(-((x - anchor[0]) ** 2 + (y - anchor[1]) ** 2) / 13.)
        image = np.full((size, size, 3), .035, np.float32)
        image += moving[..., None] * np.array([.80, .18, .12], np.float32)
        image += landmark[..., None] * np.array([.10, .48, .65], np.float32)
        pictures.append(np.clip(image, 0., 1.))
        position += velocity
        for axis in range(2):
            if position[axis] < 3 or position[axis] > size - 4:
                velocity[axis] *= -1
                position[axis] = np.clip(position[axis], 3, size - 4)
    return np.stack(pictures)


def initial_regulator():
    return {"mode": "stable", "baseline": None, "streak": 0, "remaining": 0}


def regulate(state, observed_error, base_rate):
    """Causal surprise gate: no future frames, no trainable parameters.

    Require two high-error observations before exploration; return to lower
    plasticity for consolidation. Error is from the previously issued forecast.
    """
    if not np.isfinite(observed_error) or observed_error < 0:
        raise ValueError("Prediction error must be finite and nonnegative")
    if not 0 < base_rate <= .08:
        raise ValueError("Base plasticity must lie in (0, .08]")
    state = dict(state)
    if state["baseline"] is None:
        state["baseline"] = float(observed_error)
    threshold = max(.0005, 1.6 * state["baseline"])
    if state["mode"] == "stable":
        state["streak"] = state["streak"] + 1 if observed_error > threshold else 0
        if state["streak"] >= 2:
            state.update(mode="explore", remaining=8, streak=0)
        else:
            state["baseline"] = .96 * state["baseline"] + .04 * observed_error
    else:
        state["remaining"] -= 1
        if state["remaining"] <= 0:
            if state["mode"] == "explore":
                state.update(mode="consolidate", remaining=12)
            else:
                state.update(mode="stable", remaining=0, streak=0,
                             baseline=float(observed_error))
    rates = {"stable": base_rate,
             "explore": min(.2, 2.5 * base_rate),
             "consolidate": .25 * base_rate}
    return state, float(rates[state["mode"]])


def play(frames, seed, sheet_size=8, base_rate=.04, schedule=None):
    """Prequential evaluation: score forecast, then regulate and observe."""
    if len(frames) < 3 or frames.shape[1] % sheet_size or sheet_size % 2:
        raise ValueError("Invalid frames or sheet size")
    learner = DevelopmentalSheet(sheet=sheet_size, seed=seed, plasticity=base_rate)
    initial_weights = learner.connections.copy()
    controller = initial_regulator()
    errors, rates, modes = [], [], []
    if schedule is not None and len(schedule) != len(frames):
        raise ValueError("Rate schedule must match the observation count")
    for index, frame in enumerate(frames):
        target = _pool(frame, sheet_size)
        error = float(np.mean((learner.forecast - target) ** 2))
        errors.append(error)
        if schedule is None:
            controller, rate = regulate(controller, error, base_rate)
            mode = controller["mode"]
        else:
            rate = float(schedule[index])
            if not 0 <= rate <= .2:
                raise ValueError("Invalid comparison plasticity rate")
            mode = "replay"
        # Current observation may regulate its own update, never its scored
        # forecast. The learner's local surprise also uses the issued forecast.
        learner.rate = rate
        learner.observe(frame)
        rates.append(rate)
        modes.append(mode)
    return {"errors": np.asarray(errors), "rates": np.asarray(rates),
            "modes": modes, "initial_weights": initial_weights,
            "final_weights": learner.connections.copy()}


def scores(result, segment_frames):
    errors = result["errors"]
    third = max(1, segment_frames // 3)
    parts = {}
    for index, name in enumerate(("familiar", "shifted", "return")):
        part = errors[index * segment_frames:(index + 1) * segment_frames]
        parts[name] = {"mean_mse": float(part.mean()),
                       "early_mse": float(part[:third].mean()),
                       "late_mse": float(part[-third:].mean())}
    return {"regimes": parts,
            "shift_recovery_delta": parts["shifted"]["early_mse"] - parts["shifted"]["late_mse"],
            "return_recovery_delta": parts["return"]["early_mse"] - parts["return"]["late_mse"],
            "retention_gap": parts["return"]["early_mse"] - parts["familiar"]["late_mse"],
            "mean_rate": float(result["rates"].mean()),
            "total_commanded_rate": float(result["rates"].sum()),
            "connection_change_l1": float(np.mean(np.abs(
                result["final_weights"] - result["initial_weights"]))),
            "mode_counts": {mode: result["modes"].count(mode) for mode in MODES},
            "error_curve": errors.tolist(),
            "rate_curve": result["rates"].tolist()}


def run(out="runs/regulated-plasticity", seeds=(11, 23, 41),
        segment_frames=40, size=32, sheet_size=8, base_rate=.04):
    if (not seeds or len(seeds) != len(set(seeds)) or
            segment_frames < 8 or size < 16 or size % sheet_size or sheet_size % 2 or
            not 0 < base_rate <= .08):
        raise ValueError("Invalid experiment budget")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        observations = world(int(seed) + 4000, size, segment_frames)
        adaptive = play(observations, int(seed), sheet_size, base_rate)
        schedule = adaptive["rates"]
        # Controls include the requested fixed learning rate, matched *total
        # commanded* rate, and identical rates misaligned in time.
        comparisons = {
            "adaptive": adaptive,
            "fixed": play(observations, int(seed), sheet_size, base_rate,
                          np.full(len(schedule), base_rate)),
            "matched_constant": play(observations, int(seed), sheet_size, base_rate,
                                     np.full(len(schedule), schedule.mean())),
            "shifted_schedule": play(observations, int(seed), sheet_size, base_rate,
                                     np.roll(schedule, segment_frames)),
        }
        for name, trial in comparisons.items():
            np.savez_compressed(destination / f"seed-{seed}-{name}.npz",
                                connections=trial["final_weights"])
        results.append({"seed": int(seed),
                        "arms": {name: scores(trial, segment_frames)
                                 for name, trial in comparisons.items()}})
    report = {
        "experiment": "surprise_regulated_plasticity_v1",
        "seeds": [int(s) for s in seeds],
        "segment_frames": segment_frames,
        "size": size, "sheet_size": sheet_size, "base_rate": base_rate,
        "regimes": ["familiar_motion", "motion_reversal", "familiar_motion_return"],
        "paired_inputs": True,
        "forecast_scored_before_observation": True,
        "controls": ["fixed", "matched_constant", "shifted_schedule"],
        "results": results,
        "interpretation": "Exploratory matched-seed online adaptation; no performance "
                          "claim before replicated gains on held-out worlds. A matched "
                          "total commanded rate does not imply matched weight updates. "
                          "A shifted schedule is an offline diagnostic, not an online agent."
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/regulated-plasticity")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 41])
    parser.add_argument("--segment-frames", type=int, default=40)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--sheet-size", type=int, default=8)
    parser.add_argument("--base-rate", type=float, default=.04)
    args = parser.parse_args()
    report = run(args.out, args.seeds, args.segment_frames, args.size,
                 args.sheet_size, args.base_rate)
    print(json.dumps({"seeds": report["seeds"],
                      "arms": ["adaptive", *report["controls"]]}, indent=2))


if __name__ == "__main__":
    main()
