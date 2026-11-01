"""Multi-seed robustness check for the Step 5 RGB motion NCA experiment."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.rgb_motion_nca import fit


def run(seeds=(11, 23, 41), generalization_seeds=(101, 102, 103, 104, 105),
        output="runs/rgb-motion-nca-sweep", **kwargs):
    target = Path(output)
    reports = []
    for seed in seeds:
        reports.append(fit(seed=int(seed), training_seeds=tuple(
                           int(seed) + offset for offset in range(5)),
                           generalization_seeds=generalization_seeds,
                           output=target / f"seed-{int(seed)}", **kwargs))

    def mean(section, metric):
        return float(np.mean([report[section][metric] for report in reports]))

    aggregate = {
        "experiment": "rgb_motion_nca_step5_sweep",
        "seeds": list(map(int, seeds)),
        "generalization_seeds": list(map(int, generalization_seeds)),
        "runs": len(reports),
        "overfit_mean_mse": {name: mean("overfit", name) for name in (
            "model_mse", "persistence_mse", "constant_velocity_mse")},
        "generalization_mean_mse": {name: mean("generalization", name) for name in (
            "model_mse", "persistence_mse", "constant_velocity_mse")},
        "overfit_beats_persistence_count": sum(
            report["overfit"]["model_beats_persistence"] for report in reports),
        "generalization_beats_persistence_count": sum(
            report["generalization"]["model_beats_persistence"] for report in reports),
        "generalization_beats_constant_velocity_count": sum(
            report["generalization"]["model_beats_constant_velocity"]
            for report in reports),
        "reports": reports,
        "interpretation": "A positive mean gain is descriptive; no confidence interval or "
                           "independent replication is claimed.",
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 23, 41))
    parser.add_argument("--generalization-seeds", type=int, nargs="+",
                        default=(101, 102, 103, 104, 105))
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--context", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--out", default="runs/rgb-motion-nca-sweep")
    args = parser.parse_args()
    result = run(seeds=tuple(args.seeds),
                 generalization_seeds=tuple(args.generalization_seeds),
                 output=args.out, steps=args.context + args.horizon,
                 training_steps=args.training_steps, size=args.size,
                 context=args.context, horizon=args.horizon, ticks=args.ticks,
                 channels=args.channels)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
