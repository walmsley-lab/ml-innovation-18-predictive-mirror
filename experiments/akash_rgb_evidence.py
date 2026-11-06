"""Bounded, independently scored RGB research matrix for the Akash GPU worker.

Uses existing trainers unchanged. Each model/seed/curriculum is an isolated
process with its own log and checkpoint directory; completed evidence survives
a failure of a later condition. This is NOT the older 30-run sweep.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

KIND = "rgb_evidence_v1"
MODELS = {"cnn", "transport", "nca", "nca_blend", "routing", "hierarchical"}
CURRICULA = {"linear", "mixed"}
OCCLUSION_MODELS = MODELS
FIELDS = {"kind", "seeds", "models", "curricula", "steps",
          "occlusion_models", "occlusion_steps", "context", "horizon",
          "size", "width"}


def validate(config):
    if not isinstance(config, dict) or set(config) != FIELDS or config["kind"] != KIND:
        raise ValueError("Unexpected RGB evidence manifest fields or kind")
    for key, allowed, empty in (("models", MODELS, False),
                                ("curricula", CURRICULA, False),
                                ("occlusion_models", OCCLUSION_MODELS, True)):
        values = config[key]
        if (not isinstance(values, list) or (not values and not empty)
                or len(set(values)) != len(values) or not set(values) <= allowed):
            raise ValueError("Invalid " + key)
    seeds = config["seeds"]
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Seeds must be distinct nonnegative integers")
    for name in ("steps", "occlusion_steps", "context", "horizon", "size", "width"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("Invalid " + name)
    # Existing occlusion CLI fixes 8 history, 10 future, and 32-pixel scenes.
    if config["context"] != 8 or config["horizon"] != 10 or config["size"] != 32:
        raise ValueError("Existing paired-occlusion study requires 8/10/32")
    if config["width"] < 1 or config["steps"] > 10000 or config["occlusion_steps"] > 10000:
        raise ValueError("Unsafe resource budget")
    count = len(seeds) * (len(config["models"]) * len(config["curricula"])
                          + len(config["occlusion_models"]))
    if count > 100:
        raise ValueError("Too many independent GPU conditions")
    return count


def jobs(config):
    validate(config)
    for seed in config["seeds"]:
        for model in config["models"]:
            for curriculum in config["curricula"]:
                name = f"ladder-{model}-{curriculum}-s{seed}"
                yield name, [
                    sys.executable, "-m", "experiments.rgb_world_ladder_training",
                    "--models", model, "--curricula", curriculum, "--seeds", str(seed),
                    "--steps", str(config["steps"]), "--context", str(config["context"]),
                    "--horizon", str(config["horizon"]), "--size", str(config["size"]),
                    "--width", str(config["width"]),
                ]
        for model in config["occlusion_models"]:
            name = f"occlusion-{model}-s{seed}"
            yield name, [
                sys.executable, "-m", "experiments.object_permanence_campaign",
                "--models", model, "--seeds", str(seed),
                "--steps", str(config["occlusion_steps"]),
            ]


def atomic_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, target)


def summarize_one(name, metrics):
    """Preserve field-specific scores: no cross-task or MSE-only leaderboard."""
    if name.startswith("ladder-"):
        row = metrics["results"][0]
        selected = row["test"]
        return {
            "model": row["model"], "selected_step": row["selected_step"],
            "validation_metric": row["validation_metric"],
            "occlusion_long_lead10": selected["occlusion_long"]["10"],
            "combined_lead10": selected["combined"]["10"],
            "camera_pan_action_blind_lead10": selected["camera_pan"]["10"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "limits": "Unknown camera action; low whole-image MSE alone is not object tracking.",
        }
    row = metrics["runs"][0]
    return {
        "model": row["model"], "selected_step": row["best_validation_step"],
        "reappearance_success_rate": row["test"]["success_rate"],
        "event_coverage": row["test"]["coverage"],
        "vs_persistence": row["paired_success_vs_baselines"]["persistence"],
        "vs_history_linear": row["paired_success_vs_baselines"]["history_linear"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "limits": "Paired synthetic reappearance, not general webcam object permanence.",
    }


def observation_only_baselines():
    """Exact held-out world seeds from the ladder, with no simulator-state input."""
    from experiments import rgb_world_ladder_training as ladder
    from src import rgb_benchmark as bench, rgb_event_metrics as events
    baseline = {}
    for domain in ("linear", "occlusion_long", "camera_pan", "combined"):
        records, frames = ladder.pool("test", (domain,), 4, 8, 10, 32, 2)
        forecasts = bench.controls(frames[:, :8], horizon=10)
        baseline[domain] = {
            name: events.score(predicted, records, 8, (10,))["10"]
            for name, predicted in forecasts.items()
        }
    return baseline


def run(config_path, output, max_runs, require_gpu=True):
    source = Path(config_path)
    config = json.loads(source.read_text())
    count = validate(config)
    if count > max_runs:
        raise ValueError(f"{count} conditions exceed approved max_runs={max_runs}")
    if require_gpu:
        import jax
        if not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("No JAX GPU; refusing silent CPU fallback")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": KIND, "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_revision": os.environ.get("SOURCE_REVISION", "unknown"),
        "condition_count": count, "status": "running", "conditions": [],
        "caveat": "Synthetic experiments only. No webcam generalization or calibrated uncertainty is implied.",
    }
    summary = out / "evidence.json"
    atomic_json(summary, report)
    for name, command in jobs(config):
        folder = out / "conditions" / name
        folder.mkdir(parents=True, exist_ok=True)
        command = command + ["--out", str(folder)]
        log = folder / "training.log"
        started = time.monotonic()
        environment = {**os.environ, "JAX_PLATFORMS": "cuda"}
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    env=environment, check=False)
        item = {
            "name": name, "exit_code": result.returncode,
            "duration_seconds": time.monotonic() - started,
            "log": str(log.relative_to(out)),
            "metrics": str((folder / "metrics.json").relative_to(out)),
        }
        if result.returncode == 0:
            try:
                if "observation_only_baselines" not in report:
                    report["observation_only_baselines"] = observation_only_baselines()
                item["evidence"] = summarize_one(
                    name, json.loads((folder / "metrics.json").read_text()))
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                item["error"] = "Missing/incomplete metrics: " + str(error)
        else:
            item["error"] = "Trainer returned nonzero; inspect training.log"
        report["conditions"].append(item)
        atomic_json(summary, report)
        if item.get("error"):
            report["status"] = "failed"
            atomic_json(summary, report)
            raise RuntimeError(name + ": " + item["error"])
    report["status"] = "completed"
    atomic_json(summary, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-runs", type=int, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.out, args.max_runs, args.require_gpu)
    print(json.dumps({"status": report["status"],
                      "conditions": len(report["conditions"])}))


if __name__ == "__main__":
    main()
