"""Exercise Bayesian filtering, structured memory and active observation choice."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import neuromatch_beliefs as m


def run(seed=11, replay_json=None):
    rng = np.random.default_rng(seed)
    true = 3 + 0.37 * np.arange(20)
    observations = [float(p + rng.normal(0, .18)) if not 7 <= i < 14 else None
                    for i, p in enumerate(true)]
    filtered, covariance = m.kalman_track(observations)
    last = m.hold_last(observations)
    hidden_interval = slice(7, 14)
    tracker = None
    for t in range(17):
        # Evaluator-created detections carry *appearance*, not true identity.
        detections = [(2 + .6*t, .2), (13 - .6*t, .8)]
        if 6 <= t <= 10:
            detections = detections[1:]  # temporary occlusion of first target
        if t > 0 and t % 2:
            detections = detections[::-1]
        tracker = m.update_object_memory(tracker, detections, t)
    actions = {
        "informative_view": np.array([[.95, .05], [.05, .95]]),
        "weak_view": np.array([[.55, .45], [.45, .55]]),
    }
    choice, gains = m.select_active_view([.5, .5], actions)
    result = {
        "seed": seed,
        "filter": {
            "occluded_kalman_mae": float(np.abs(filtered[hidden_interval, 0] - true[hidden_interval]).mean()),
            "occluded_persistence_mae": float(np.abs(last[hidden_interval] - true[hidden_interval]).mean()),
            "final_position_variance": float(covariance[-1, 0, 0]),
        },
        "structured_memory": {
            "final_slot_positions": [float(t["position"]) for t in tracker],
            "final_abs_position_errors": [
                abs(tracker[0]["position"] - (2 + .6*16)),
                abs(tracker[1]["position"] - (13 - .6*16))],
            "note": "Two appearance-tagged 1D detections; not learned detection from RGB."
        },
        "active_observation": {
            "selected": choice, "expected_information_nats": gains,
            "uniform_action_expected_information_nats": float(np.mean(list(gains.values()))),
            "note": "Analytic known-sensor model; no learned active gaze policy."
        },
    }
    if replay_json:
        result["replay_retention"] = m.replay_retention(json.loads(Path(replay_json).read_text()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--replay-json", default=None,
                        help="Existing roadmap_33_rgb_replay metrics.json")
    parser.add_argument("--out", default="runs/neuromatch-beliefs/metrics.json")
    args = parser.parse_args()
    result = run(args.seed, args.replay_json)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
