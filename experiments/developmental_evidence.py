"""Strict, resumable developmental evidence campaign for Akash GPU jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import _pool
from experiments.developmental_transfer import _object_scores, _wave_data
from src import developmental_jax as model, physical_fields, world3d

KIND = "developmental_evidence_v1"
ARMS = ("coherent", "shuffled", "uncorrelated", "no_wave")
FIELDS = {"kind", "seeds", "arms", "pretrain_updates", "visual_updates",
          "active_3d_updates", "size", "sheet_size", "checkpoint_every",
          "plasticity", "freeze_wave_comparison", "shuffle_action_comparison"}


def validate(config):
    if not isinstance(config, dict) or set(config) != FIELDS or config.get("kind") != KIND:
        raise ValueError("Unexpected developmental evidence manifest fields or kind")
    seeds = config["seeds"]
    if (not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError("Seeds must be distinct nonnegative integers")
    arms = config["arms"]
    if arms != list(ARMS):
        raise ValueError("Developmental arms must be coherent, shuffled, uncorrelated, no_wave")
    for key in ("pretrain_updates", "visual_updates", "active_3d_updates", "size",
                "sheet_size", "checkpoint_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["pretrain_updates"] < config["checkpoint_every"]:
        raise ValueError("Pretraining must include a checkpoint")
    if config["size"] < 16 or config["size"] % config["sheet_size"] or config["sheet_size"] % 2:
        raise ValueError("Size must be divisible by an even sheet_size")
    if type(config["plasticity"]) not in (int, float) or not 0 <= config["plasticity"] <= .2:
        raise ValueError("Invalid plasticity")
    if config["freeze_wave_comparison"] is not True or config["shuffle_action_comparison"] is not True:
        raise ValueError("Required frozen-connectivity and shuffled-action comparisons cannot be disabled")
    count = len(seeds) * len(ARMS)
    if count > 20:
        raise ValueError("Developmental campaign exceeds 20 conditions")
    return count


def jobs(config):
    count = validate(config)
    for seed in config["seeds"]:
        for arm in config["arms"]:
            yield f"{arm}-s{seed}", ["developmental", str(seed), arm]


def _world(seed, updates, size, domain):
    record = world3d.episode(seed, updates, size, domain)
    return record["rgb"], record["camera_actions"], record


def _save(path, state, metadata):
    payload = {key: np.asarray(value) for key, value in state.items()}
    payload["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **payload)


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in
                 ("phase", "hidden", "trace", "homeostasis", "connections",
                  "previous", "forecast", "plasticity")}
    import jax.numpy as jnp
    return {key: jnp.asarray(value) for key, value in state.items()}


def _pretraining(seed, arm, updates, size):
    if arm == "coherent":
        return _wave_data(seed, updates, size, "coherent"), np.zeros((updates - 1, 2))
    if arm == "shuffled":
        return _wave_data(seed, updates, size, "shuffled"), np.zeros((updates - 1, 2))
    if arm == "uncorrelated":
        return _wave_data(seed, updates, size, "uncorrelated"), np.zeros((updates - 1, 2))
    return _wave_data(seed, updates, size, "none"), np.zeros((updates - 1, 2))


def _run_sequence(state, frames, actions, step_fn, errors, ious,
                  checkpoint_dir=None, stage=None, checkpoint_every=0,
                  seed=None, arm=None):
    predictions = []
    for index, frame in enumerate(frames):
        prediction = model.forecast(state)
        target = _pool(frame, state["phase"].shape[0])
        errors.append(float(np.mean((np.asarray(prediction) - target) ** 2)))
        pred_mask = np.mean(np.asarray(prediction), axis=-1) > .08
        true_mask = np.mean(target, axis=-1) > .08
        union = np.logical_or(pred_mask, true_mask).sum()
        ious.append(float(np.logical_and(pred_mask, true_mask).sum() / union) if union else 1.)
        predictions.append(np.asarray(prediction))
        action = actions[index - 1] if index and index - 1 < len(actions) else (0., 0.)
        state, _ = step_fn(state, frame, action)
        if checkpoint_dir and checkpoint_every and ((index + 1) % checkpoint_every == 0
                                                    or index + 1 == len(frames)):
            _save(Path(checkpoint_dir) / f"checkpoint-{stage}-{index + 1}.npz", state,
                  {"stage": stage, "update": index + 1, "seed": seed, "arm": arm})
    return state, predictions


def run(config_path, output, max_runs, require_gpu=True):
    source = Path(config_path)
    config = json.loads(source.read_text())
    count = validate(config)
    if count > max_runs:
        raise ValueError(f"{count} conditions exceed approved max_runs={max_runs}")
    import jax
    if require_gpu and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("No JAX GPU; refusing CPU fallback")
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    report_path = out / "evidence.json"
    report = {"kind": KIND, "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "source_revision": os.environ.get("SOURCE_REVISION", "unknown"),
              "condition_count": count, "status": "running", "conditions": [],
              "gpu_required": bool(require_gpu)}
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("config_sha256") != report["config_sha256"]:
            raise ValueError("Existing evidence report has a different manifest")
        report = prior
    import jax.numpy as jnp
    step_fn = __import__("jax").jit(model.step)
    for name, _ in jobs(config):
        arm = name.rsplit("-s", 1)[0]
        if any(item.get("name") == name and item.get("exit_code") == 0
               for item in report["conditions"]):
            continue
        seed = int(name.rsplit("s", 1)[1])
        condition = out / "conditions" / name
        condition.mkdir(parents=True, exist_ok=True)
        state = model.init(config["sheet_size"], 7, seed, config["plasticity"])
        errors, ious = [], []
        pre_frames, pre_actions = _pretraining(seed, arm, config["pretrain_updates"], config["size"])
        state, _ = _run_sequence(state, pre_frames, pre_actions, step_fn, errors, ious,
                                 condition, "pretraining", config["checkpoint_every"], seed, arm)
        _save(condition / "checkpoint-pretraining.npz", state,
              {"stage": "pretraining", "seed": seed, "arm": arm})
        visual, visual_actions, visual_record = _world(seed + 1000, config["visual_updates"], config["size"], "train")
        pre_visual = len(errors)
        state, visual_predictions = _run_sequence(state, visual, visual_actions, step_fn, errors, ious,
                                                  condition, "visual", config["checkpoint_every"], seed, arm)
        _save(condition / "checkpoint-visual.npz", state,
              {"stage": "visual", "seed": seed, "arm": arm})
        active, active_actions, active_record = _world(seed + 2000, config["active_3d_updates"], config["size"], "ood_camera")
        active_start = len(errors)
        state, active_predictions = _run_sequence(state, active, active_actions, step_fn, errors, ious,
                                                  condition, "active-3d", config["checkpoint_every"], seed, arm)
        _save(condition / "checkpoint-active-3d.npz", state,
              {"stage": "active_3d", "seed": seed, "arm": arm})
        # Paired controls are diagnostic runs from the same visual checkpoint.
        frozen = _load(condition / "checkpoint-pretraining.npz")
        frozen_fn = __import__("jax").jit(lambda s, f, a: model.step(s, f, a, False))
        frozen, frozen_predictions = _run_sequence(frozen, visual, visual_actions, frozen_fn, [], [])
        shuffled_actions = np.asarray(visual_actions).copy()
        np.random.default_rng(seed + 9000).shuffle(shuffled_actions, axis=0)
        shuffled_state = _load(condition / "checkpoint-visual.npz")
        shuffled_errors = []
        shuffled_state, _ = _run_sequence(shuffled_state, active, shuffled_actions,
                                          step_fn, shuffled_errors, [])
        payload = {"name": name, "exit_code": 0,
                   "metrics": {"pretraining_mse": float(np.mean(errors[:pre_visual])),
                               "visual_mse_auc": float(np.mean(errors[pre_visual:active_start])),
                               "active_3d_mse": float(np.mean(errors[active_start:])),
                               "visual_foreground_iou": float(np.mean(ious[pre_visual:active_start])),
                               "active_foreground_iou": float(np.mean(ious[active_start:])),
                               "visual_object_metrics": _object_scores(
                                   visual_predictions, visual_record),
                               "active_object_metrics": _object_scores(
                                   active_predictions, active_record),
                               "frozen_visual_mse": float(np.mean([np.mean((p - _pool(f, config["sheet_size"])) ** 2) for p, f in zip(frozen_predictions, visual)])),
                               "shuffled_action_active_mse": float(np.mean(shuffled_errors)),
                               "checkpoint_files": ["checkpoint-pretraining.npz", "checkpoint-visual.npz", "checkpoint-active-3d.npz"]}}
        report["conditions"] = [item for item in report["conditions"] if item.get("name") != name]
        report["conditions"].append(payload)
        report["conditions"].sort(key=lambda item: item["name"])
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        injected = os.environ.get("DEVELOPMENTAL_INJECT_FAILURE_AFTER")
        if injected and len(report["conditions"]) >= int(injected):
            raise RuntimeError("Injected developmental condition failure")
    report["status"] = "completed"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-runs", type=int, required=True)
    p.add_argument("--require-gpu", action="store_true")
    a = p.parse_args()
    print(json.dumps({"status": run(a.config, a.out, a.max_runs, a.require_gpu)["status"]}))
