"""Evaluate observation-only history baselines on paired occlusion worlds."""
import argparse
import json
from pathlib import Path
import numpy as np

from src import paired_occlusion_histories as world
from src import history_baselines as hist
from src import reappearance_scoring as events


def _success(scores, location_tolerance=3.0):
    result = []
    for row in scores["results"]:
        for e in row["events"]:
            ok = (e["reappearance_centroid_error_px"] is not None
                  and e["reappearance_centroid_error_px"] <= location_tolerance
                  and e["timing_error_frames"] is not None
                  and abs(e["timing_error_frames"]) <= 1
                  and e["premature_visible_frames"] == 0)
            result.append(int(ok))
    return result


def run(seeds=(11,23,41), out="runs/history-baselines/metrics.json",
        context=8, horizon=10):
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Independent scene seed pairs must be unique")
    records = [item for seed in seeds for item in world.pair(int(seed),context,horizon)]
    prefix = np.stack([r["frames"][:context] for r in records])
    future = np.stack([r["frames"][context:context+horizon] for r in records])
    methods = {
        "persistence": lambda x: np.repeat(x[-1:], horizon, axis=0),
        "rgb_history_linear": lambda x: hist.history_linear(x, horizon),
        "rgb_history_kalman": lambda x: hist.kalman_history(x, horizon),
        "known_prior_linear_reference": lambda x: world.linear_history_forecast(x, horizon)
    }
    report = {"protocol": "paired-history-observation-baselines-v1",
              "seeds": list(map(int,seeds)), "context": context,
              "horizon": horizon, "models": {}}
    for name, fn in methods.items():
        predicted = np.stack([fn(x) for x in prefix])
        score = events.score(predicted, records, context)
        hits = np.asarray(_success(score), float).reshape(-1, 2)
        contrast_truth = future[::2] - future[1::2]
        contrast_pred = predicted[::2] - predicted[1::2]
        report["models"][name] = {
            "input_tier": "declared_known_prior" if name == "known_prior_linear_reference" else "rgb_only",
            "pair_success": hits.mean(axis=1).tolist(),
            "success_rate": float(hits.mean()),
            "history_contrast_mse": float(np.mean((contrast_pred - contrast_truth)**2)),
            "rgb_mse": float(np.mean((predicted - future)**2)),
            "event_scoring": score
        }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/history-baselines/metrics.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,23,41])
    a = parser.parse_args()
    r = run(a.seeds,a.out)
    print(json.dumps({name: {"success": value["success_rate"],
                              "tier": value["input_tier"]}
                      for name,value in r["models"].items()},indent=2))


if __name__=="__main__":
    main()
