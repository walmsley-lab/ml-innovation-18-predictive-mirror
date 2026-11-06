"""Complete small developmental training comparison with persistent checkpoints.

The developmental arm sees waves, then visual 3D experience, then active-camera
3D experience.  The matched control receives the same number of observations
but skips waves.  Both retain one sheet and are evaluated on held-out 3D RGB.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import DevelopmentalSheet, _pool, _retinal_frames
from src import world3d


def _state(sheet):
    return {"phase": sheet.phase, "hidden": sheet.hidden,
            "trace": sheet.trace, "homeostasis": sheet.homeostasis,
            "connections": sheet.connections, "previous": sheet.previous,
            "forecast": sheet.forecast}


def _save(sheet, path, stage, seed):
    payload = {key: value for key, value in _state(sheet).items()}
    payload.update(stage=np.asarray(stage), seed=np.asarray(seed))
    np.savez_compressed(path, **payload)


def _load(path, sheet_size, seed):
    sheet = DevelopmentalSheet(sheet_size, 7, seed, .04)
    with np.load(path, allow_pickle=False) as archive:
        for key in _state(sheet):
            setattr(sheet, key, archive[key].copy())
    return sheet


def _frames(seed, frames, size, domain):
    record = world3d.episode(seed, frames, size, domain)
    return record["rgb"], record["camera_actions"]


def _forecast(sheet):
    return sheet.forecast


def _present(sheet, frames, actions, errors):
    for index, frame in enumerate(frames):
        before = _forecast(sheet)
        errors.append(float(np.mean((before - _pool(frame, sheet.sheet)) ** 2)))
        action = actions[index - 1] if index and index - 1 < len(actions) else (0., 0.)
        sheet.observe(frame, action)


def _evaluate(sheet, seed, frames, size):
    observations, actions = _frames(seed, frames, size, "ood_depth")
    errors = []
    _present(sheet, observations, actions, errors)
    return {"mean_mse": float(np.mean(errors)), "last_mse": float(errors[-1]),
            "first_mse": float(errors[0])}


def _locality(sheet):
    # Nonzero spatial variation confirms weights are not one global matrix.
    return float(np.mean(np.std(sheet.connections, axis=(0, 1))))


def run(out="runs/developmental-training", seed=11, size=16, sheet_size=4,
        stage_frames=8):
    if stage_frames < 3 or size < 16 or size % sheet_size or sheet_size % 2:
        raise ValueError("Need stage_frames>=3, divisible size and even sheet")
    dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    wave, _ = _retinal_frames("wave", seed, stage_frames, size)
    visual, visual_actions = _frames(seed + 10, stage_frames, size, "train")
    active, active_actions = _frames(seed + 20, stage_frames, size, "ood_camera")

    arms = {"developmental": DevelopmentalSheet(sheet_size, 7, seed, .04),
            "no_wave": DevelopmentalSheet(sheet_size, 7, seed, .04)}
    errors = {name: [] for name in arms}
    stages = {
        "developmental": ((wave, np.zeros((stage_frames - 1, 2))),
                           (visual, visual_actions), (active, active_actions)),
        "no_wave": ((visual, visual_actions), (visual, visual_actions),
                     (active, active_actions)),
    }
    checkpoints = {}
    for name, sheet in arms.items():
        checkpoints[name] = []
        for stage, (frames, actions) in zip(("waves", "visual", "active_3d"), stages[name]):
            _present(sheet, frames, actions, errors[name])
            checkpoint = dest / f"{name}-{stage}.npz"
            _save(sheet, checkpoint, stage, seed)
            checkpoints[name].append(str(checkpoint))

    report = {"experiment": "developmental_training_comparison_v1",
              "seed": seed, "stage_frames": stage_frames,
              "total_updates": stage_frames * 3,
              "same_initial_state": True, "same_update_budget": True,
              "arms": {}, "evaluation": {}}
    for name, sheet in arms.items():
        report["arms"][name] = {
            "training_mse_first": errors[name][0],
            "training_mse_last": errors[name][-1],
            "checkpoint_stages": checkpoints[name],
            "final_connection_locality": _locality(sheet),
            "final_metrics": sheet.metrics(),
        }
        report["evaluation"][name] = _evaluate(
            _load(checkpoints[name][-1], sheet_size, seed), seed + 100,
            stage_frames, size)
    report["decision_rule"] = (
        "Invest only if developmental evaluation MSE beats no_wave and local "
        "weights remain spatially varied without collapse; otherwise extend "
        "the small study before a larger 3D/webcam campaign.")
    (dest / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/developmental-training")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--sheet-size", type=int, default=4)
    p.add_argument("--stage-frames", type=int, default=8)
    a = p.parse_args()
    print(json.dumps(run(a.out, a.seed, a.size, a.sheet_size, a.stage_frames), indent=2))
