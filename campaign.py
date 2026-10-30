"""Freeze and inspect an experiment matrix; training requires explicit approval."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from sweep import run


def validate(config):
    required = {"seeds", "architectures", "curricula", "steps", "batch_size",
                "episodes", "checkpoint_every", "loss_mode", "primary_metric", "replay_fraction"}
    if set(config) != required:
        raise ValueError("Campaign fields changed")
    for name, allowed in (("architectures", {"recurrent", "feedforward", "multisheet"}),
                          ("curricula", {"baseline", "staged", "shuffled"})):
        values = config[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(f"Invalid {name}")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    for name in ("steps", "batch_size", "episodes", "checkpoint_every"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if config["loss_mode"] not in ("plain", "contrast") or config["replay_fraction"] != 0.10:
        raise ValueError("Loss or replay fraction differs from frozen sweep")
    if config["primary_metric"] != "reappearance_location_mae_px":
        raise ValueError("Primary metric changed")
    return len(seeds) * len(config["architectures"]) * len(config["curricula"])


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan(config):
    count = validate(config)
    return {"campaign_sha256": digest(config), "runs": count,
            "updates": count * config["steps"],
            "examples": count * config["steps"] * config["batch_size"],
            "note": "Does not estimate lease cost or imply training has run"}


def execute(config, output, max_runs, require_gpu=False):
    count = validate(config)
    if count > max_runs:
        raise ValueError(f"Refusing {count} conditions (max_runs={max_runs})")
    output = Path(output)
    frozen = output / "campaign.json"
    signature = {"configuration": config, "sha256": digest(config),
                 "source_revision": os.getenv("SOURCE_REVISION", "unknown")}
    if frozen.exists():
        if json.loads(frozen.read_text()) != signature:
            raise ValueError("Campaign or source revision changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Untracked campaign directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        part = output / "campaign.json.part"
        part.write_text(json.dumps(signature, indent=2) + "\n")
        os.replace(part, frozen)
    return run(output, config["seeds"], config["architectures"], config["curricula"],
               config["steps"], config["batch_size"], config["episodes"],
               require_gpu, config["loss_mode"], resume=True,
               checkpoint_every=config["checkpoint_every"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/campaign.json")
    parser.add_argument("--output", default="runs/campaign")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    print(json.dumps(plan(config), indent=2), flush=True)
    if args.execute:
        execute(config, args.output, args.max_runs, args.require_gpu)


if __name__ == "__main__":
    main()
