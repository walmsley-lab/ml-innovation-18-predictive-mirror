"""Validate CPU integration artifacts before anyone pays for GPU time."""

import argparse
import json
from pathlib import Path


def check(root, seeds, architectures, curricula, loss_mode="plain"):
    """Return complete conditions; refuse missing/partial or inconsistent artifacts."""
    root = Path(root)
    rows = json.loads((root / "runs.json").read_text())
    expected = {(architecture, curriculum, seed)
                for architecture in architectures for curriculum in curricula for seed in seeds}
    observed = {(row["architecture"], row["curriculum"], row["seed"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("Run matrix incomplete or duplicated")
    summary = json.loads((root / "summary.json").read_text())
    if len(summary) != len(architectures) * len(curricula):
        raise ValueError("Summary missing experimental cells")
    for architecture, curriculum, seed in sorted(expected):
        path = root / f"{architecture}-{curriculum}-seed{seed}"
        spec = json.loads((path / "spec.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        if (spec["architecture"], spec["curriculum"], spec["seed"], spec["loss_mode"]) != (
                architecture, curriculum, seed, loss_mode):
            raise ValueError(f"Incorrect config: {path}")
        if metrics["architecture"] != architecture or metrics["seed"] != seed:
            raise ValueError(f"Incorrect trained artifact: {path}")
        for name in ("checkpoint.npz", "parameters.npz", "evaluation.json"):
            if not (path / name).is_file():
                raise ValueError(f"Missing artifact: {path / name}")
    return {"n": len(rows), "conditions": sorted(expected), "status": "CPU integration only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--architectures", nargs="+", required=True)
    parser.add_argument("--curricula", nargs="+", required=True)
    parser.add_argument("--loss-mode", default="plain")
    args = parser.parse_args()
    report = check(args.root, args.seeds, args.architectures, args.curricula, args.loss_mode)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
