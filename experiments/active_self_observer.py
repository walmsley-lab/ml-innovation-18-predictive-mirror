"""An active perceptual observer with inspectable beliefs and bounded vision.

The observer must choose a fovea before receiving a frame. A handcrafted
red-object detector updates a short-term motion belief; the existing local
developmental sheet learns from the same limited RGB observation. Reported
"confidence" is a heuristic, not a calibrated probability or subjective state.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import DevelopmentalSheet, _pool
from experiments.regulated_plasticity import initial_regulator, regulate, world

POLICIES = ("active", "center", "raster", "active_no_memory")
BACKGROUND = np.float32(.035)


def red_center(frame):
    """Hand-engineered retinal feature, not a learned object detector."""
    signal = np.maximum(frame[..., 0] - frame[..., 1] - .07, 0.)
    mass = float(signal.sum())
    if mass < .25:
        return None
    yy, xx = np.indices(signal.shape)
    return np.asarray([(xx * signal).sum() / mass,
                       (yy * signal).sum() / mass], np.float64)


def foveate(frame, gaze, radius):
    """Return exactly one circular observation and its observed-pixel mask."""
    yy, xx = np.indices(frame.shape[:2])
    visible = (xx - gaze[0]) ** 2 + (yy - gaze[1]) ** 2 <= radius ** 2
    glimpse = np.where(visible[..., None], frame, BACKGROUND)
    return glimpse.astype(np.float32), visible


def initial_belief(size):
    return {"position": np.asarray([size * .4, size * .55], np.float64),
            "velocity": np.zeros(2, np.float64), "last_seen": None,
            "last_seen_step": None, "misses": 0, "confidence": 0.25}


def expected_position(belief, size):
    """One-step extrapolation from observations, never from world state."""
    return np.clip(belief["position"] + belief["velocity"], 0., size - 1.)


def choose_gaze(belief, t, policy, size, radius):
    """Decide BEFORE seeing frame[t]; all policies receive one equal aperture."""
    if policy not in POLICIES:
        raise ValueError("Unknown gaze policy")
    if policy in ("active", "active_no_memory"):
        predicted = expected_position(belief, size)
        if belief["misses"] > 1:
            # Bounded search around our OWN prediction, not an oracle location.
            offset = np.asarray(((0., 0.), (radius, 0.), (0., radius),
                                 (-radius, 0.), (0., -radius)))[
                                     (belief["misses"] - 2) % 5]
            predicted += offset
        gaze = predicted
    elif policy == "center":
        gaze = np.asarray([size / 2, size / 2])
    else:
        # Fixed, stimulus-independent raster; same one-glimpse sensor budget.
        locations = ((.25, .25), (.75, .25), (.75, .75), (.25, .75))
        gaze = np.asarray(locations[t % 4]) * (size - 1)
    return np.clip(gaze, radius, size - radius - 1)


def update_belief(belief, observation, t, size, radius):
    """Causal tracker: correct predicted state when the object is actually seen."""
    state = dict(belief)
    prior = expected_position(state, size)
    if observation is None:
        state["position"] = prior
        state["misses"] += 1
        state["confidence"] *= .75
        return state, None
    innovation = float(np.linalg.norm(observation - prior))
    if state["last_seen"] is not None:
        dt = t - state["last_seen_step"]
        velocity = (observation - state["last_seen"]) / dt
        state["velocity"] = .35 * state["velocity"] + .65 * velocity
    state["position"] = observation
    state["last_seen"] = observation.copy()
    state["last_seen_step"] = t
    state["misses"] = 0
    state["confidence"] = float(np.clip(
        .65 * state["confidence"] + .35 * np.exp(-innovation / radius), 0., 1.))
    return state, innovation


def observations(seed, size, segment_frames):
    """Evaluator has unoccluded truth; model receives only the occluded scene."""
    raw = world(seed, size, segment_frames)
    seen = raw.copy()
    start = segment_frames + segment_frames // 3
    stop = start + max(2, segment_frames // 6)
    occluded = np.zeros(len(raw), bool)
    for t in range(start, min(stop, 2 * segment_frames)):
        # Hide the moving red object without revealing its position to agent.
        signal = raw[t, ..., 0] - raw[t, ..., 1] > .07
        seen[t, signal] = BACKGROUND
        occluded[t] = True
    return seen, raw, occluded


def episode(frames, truth_frames, occluded, seed, policy, radius, sheet_size):
    if policy not in POLICIES or not len(frames) == len(truth_frames) == len(occluded):
        raise ValueError("Mismatched episode or policy")
    size = frames.shape[1]
    learner = DevelopmentalSheet(sheet_size, 7, seed, .04)
    regulator = initial_regulator()
    belief = initial_belief(size)
    last_gaze = np.asarray([size * .4, size * .55])
    log = []
    for t, frame in enumerate(frames):
        prior_belief = initial_belief(size) if policy == "active_no_memory" else belief
        gaze = choose_gaze(prior_belief, t, policy, size, radius)
        expected = expected_position(prior_belief, size)
        # One action-conditioned observation; no peeking at full/current frame.
        glimpse, pixels = foveate(frame, gaze, radius)
        detected = red_center(glimpse)
        target = _pool(glimpse, sheet_size)
        error = float(np.mean((learner.forecast - target) ** 2))
        regulator, rate = regulate(regulator, error, .04)
        learner.rate = rate
        learner.observe(glimpse, (gaze - last_gaze) / size)
        belief, innovation = update_belief(prior_belief, detected, t, size, radius)
        last_gaze = gaze
        # Evaluator-only truth; never enters choice, belief, or model update.
        actual = red_center(truth_frames[t])
        position_error = (float(np.linalg.norm(expected - actual))
                          if actual is not None else None)
        log.append({
            "t": t,
            "observer": {
                "gaze_xy": gaze.tolist(), "expected_xy": expected.tolist(),
                "belief_xy": belief["position"].tolist(),
                "confidence_proxy": belief["confidence"],
                "innovation_pixels": innovation,
                "object_detected": detected is not None,
                "issued_forecast_mse_on_glimpse": error,
                "plasticity_mode": regulator["mode"], "plasticity_rate": rate,
            },
            "evaluation": {"true_xy": None if actual is None else actual.tolist(),
                           "tracking_error_pixels": position_error,
                           "source_occluded": bool(occluded[t]),
                           "aperture_pixels": int(pixels.sum())},
        })
    track = np.asarray([x["evaluation"]["tracking_error_pixels"] for x in log], float)
    detected = np.asarray([x["observer"]["object_detected"] for x in log])
    n = len(log) // 3
    return {"policy": policy, "seed": int(seed),
            "mean_tracking_error_px": float(np.nanmean(track)),
            "shift_tracking_error_px": float(np.nanmean(track[n:2*n])),
            "return_tracking_error_px": float(np.nanmean(track[2*n:])),
            "detected_fraction": float(detected.mean()),
            "occluded_detection_fraction": float(detected[occluded].mean())
                if np.any(occluded) else None,
            "regulator_modes": {mode: sum(x["observer"]["plasticity_mode"] == mode
                                          for x in log)
                                for mode in ("stable", "explore", "consolidate")},
            "final_connection_l1": learner.metrics()["connection_l1"],
            "trace": log}


def run(out="runs/active-self-observer", seeds=(11, 23, 41),
        size=32, segment_frames=40, radius=6, sheet_size=8):
    if (not seeds or len(set(seeds)) != len(seeds) or size < 16
            or segment_frames < 8 or radius < 3 or 2 * radius >= size - 2
            or size % sheet_size or sheet_size % 2):
        raise ValueError("Invalid experiment budget")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        frames, truth, occluded = observations(int(seed) + 4000, size, segment_frames)
        arms = {}
        for policy in POLICIES:
            result = episode(frames, truth, occluded, int(seed), policy,
                             radius, sheet_size)
            trace = destination / f"seed-{seed}-{policy}.jsonl"
            trace.write_text("".join(json.dumps(t) + "\n" for t in result.pop("trace")))
            result["trace_file"] = trace.name
            arms[policy] = result
        rows.append({"seed": int(seed), "arms": arms})
    report = {
        "experiment": "active_perceptual_self_monitoring_v1",
        "seeds": list(map(int, seeds)), "segment_frames": segment_frames,
        "size": size, "radius": radius, "sheet_size": sheet_size,
        "policies": list(POLICIES), "same_one_glimpse_per_frame": True,
        "all_choices_before_current_observation": True,
        "results": rows,
        "limitations": "Handcrafted red-object detector and confidence heuristic; "
                       "not a learned perception head, calibrated belief, conscious "
                       "experience or demonstration of subjective qualia. "
                       "Observer cannot read evaluator truth. Results should be "
                       "interpreted against fixed/raster and memory-ablated controls."
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/active-self-observer")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 41])
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--segment-frames", type=int, default=40)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--sheet-size", type=int, default=8)
    a = parser.parse_args()
    results = run(a.out, tuple(a.seeds), a.size, a.segment_frames,
                  a.radius, a.sheet_size)
    print(json.dumps({"seeds": results["seeds"],
                      "policies": results["policies"], "out": a.out}, indent=2))


if __name__ == "__main__":
    main()
