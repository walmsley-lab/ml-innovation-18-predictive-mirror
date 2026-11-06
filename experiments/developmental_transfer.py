"""Developmental transfer: do coherent waves improve later visual learning?

All arms receive identical visual and 3D sequences after pretraining.  The
forecast is captured before each observation is absorbed, so every score is
against an observation that was not available when that forecast was issued.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import DevelopmentalSheet, _pool
from experiments.developmental_training import _save
from src import physical_fields, world3d


SEEDS = (11, 23, 41, 47, 53)
ARMS = ("A_coherent", "B_shuffled", "C_uncorrelated", "D_no_wave")


def _wave_data(seed, updates, size, mode):
    wave = physical_fields.episode(seed, "wave", size, updates)["observed"]
    rgb = np.repeat(wave[..., None], 3, axis=-1)
    rgb = (rgb - rgb.min()) / max(1e-6, float(rgb.max() - rgb.min()))
    if mode == "coherent":
        return rgb.astype(np.float32)
    rng = np.random.default_rng(seed + 700)
    if mode == "shuffled":
        flat = rgb.reshape(updates, -1, 3)
        return np.stack([flat[i][rng.permutation(flat.shape[1])]
                         for i in range(updates)]).reshape(rgb.shape).astype(np.float32)
    if mode == "uncorrelated":
        return rng.random(rgb.shape, dtype=np.float32)
    if mode == "none":
        return np.zeros_like(rgb, dtype=np.float32)
    raise ValueError("Unknown pretraining mode")


def _visual_data(seed, updates, size, domain):
    record = world3d.episode(seed, updates, size, domain)
    return record["rgb"], record["camera_actions"], record


def _forecast(sheet):
    return sheet.forecast


def _score_prediction(prediction, target):
    pred = np.mean(prediction, axis=-1) > .08
    truth = np.mean(target, axis=-1) > .08
    union = np.logical_or(pred, truth).sum()
    intersection = np.logical_and(pred, truth).sum()
    return float(intersection / union) if union else 1.0


def _object_scores(predictions, record):
    """Measure visible localization and persistence at projected object centers."""
    visible_hits, visible_total, hidden_hits, hidden_total = 0, 0, 0, 0
    for tick, prediction in enumerate(predictions):
        pred = np.mean(prediction, axis=-1)
        h, w = pred.shape
        uv, depth = world3d.project(record["centers_world"][tick],
                                    record["T_world_camera"][tick][:3, 3],
                                    np.arctan2(record["T_world_camera"][tick][0, 2],
                                               record["T_world_camera"][tick][0, 0]),
                                    record["K"])
        full = record["rgb"].shape[1]
        for object_id, (point, z) in enumerate(zip(uv, depth)):
            full_x, full_y = np.rint(point).astype(int)
            x, y = np.rint(point * (w / full)).astype(int)
            if z <= 0 or not (0 <= x < w and 0 <= y < h and
                              0 <= full_x < full and 0 <= full_y < full):
                continue
            active = bool(pred[y, x] > .08)
            # instance_id distinguishes visible surfaces from objects hidden by
            # another sphere; both cases are evaluated at the predicted location.
            is_visible = int(record["instance_id"][tick, full_y, full_x]) == object_id
            if is_visible:
                visible_total += 1
                visible_hits += int(active)
            else:
                hidden_total += 1
                hidden_hits += int(active)
    return {
        "visible_localization_rate": visible_hits / visible_total if visible_total else 0.,
        "occluded_persistence_rate": hidden_hits / hidden_total if hidden_total else 0.,
        "visible_points": visible_total, "occluded_points": hidden_total,
    }


def _learn(sheet, frames, actions, errors, iou):
    predictions = []
    for index, frame in enumerate(frames):
        prediction = _forecast(sheet)
        errors.append(float(np.mean((prediction - _pool(frame, sheet.sheet)) ** 2)))
        iou.append(_score_prediction(prediction, _pool(frame, sheet.sheet)))
        predictions.append(prediction)
        action = actions[index - 1] if index and index - 1 < len(actions) else (0., 0.)
        sheet.observe(frame, action)
    return predictions


def _arm(seed, arm, out, pretrain_updates, visual_updates, size, sheet_size,
         freeze_wave=False):
    mode = {"A_coherent": "coherent", "A_coherent_frozen": "coherent",
            "B_shuffled": "shuffled", "C_uncorrelated": "uncorrelated",
            "D_no_wave": "none"}[arm]
    sheet = DevelopmentalSheet(sheet_size, 7, seed, .04)
    errors, iou, checkpoints = [], [], []
    _learn(sheet, _wave_data(seed, pretrain_updates, size, mode),
           np.zeros((pretrain_updates - 1, 2)), errors, iou)
    pre_path = Path(out) / f"seed-{seed}-{arm}-pretraining.npz"
    _save(sheet, pre_path, "pretraining", seed)
    pre_connections = sheet.connections.copy()
    visual, visual_actions, _ = _visual_data(seed + 1000, visual_updates, size, "train")
    if freeze_wave:
        sheet.rate = 0.
    visual_predictions = _learn(sheet, visual, visual_actions, errors, iou)
    visual_path = Path(out) / f"seed-{seed}-{arm}-visual.npz"
    _save(sheet, visual_path, "visual", seed)
    active, active_actions, active_record = _visual_data(
        seed + 2000, visual_updates, size, "ood_camera")
    active_predictions = _learn(sheet, active, active_actions, errors, iou)
    active_path = Path(out) / f"seed-{seed}-{arm}-active-3d.npz"
    _save(sheet, active_path, "active_3d", seed)
    local_delta = float(np.mean(np.abs(sheet.connections - pre_connections)))
    speed_window = max(1, visual_updates // 4)
    metrics = {
        "seed": seed, "arm": arm, "freeze_wave_connections": freeze_wave,
        "learning_speed": {
            "visual_first_quarter_mse": float(np.mean(errors[pretrain_updates:pretrain_updates + speed_window])),
            "visual_last_quarter_mse": float(np.mean(errors[pretrain_updates + visual_updates - speed_window:pretrain_updates + visual_updates])),
            "visual_mse_auc": float(np.mean(errors[pretrain_updates:pretrain_updates + visual_updates])),
        },
        "object_localization": _object_scores(visual_predictions, _visual_data(seed + 1000, visual_updates, size, "train")[2]),
        "active_3d": {**_object_scores(active_predictions, active_record),
                      "camera_generalization_mse": float(np.mean(errors[-visual_updates:]))},
        "foreground_iou": float(np.mean(iou[-visual_updates:])),
        "pre_to_post_connection_delta": local_delta,
        "final_sheet": sheet.metrics(),
        "checkpoints": [str(pre_path), str(visual_path), str(active_path)],
    }
    return metrics


def run(out="runs/developmental-transfer", seeds=SEEDS, pretrain_updates=120,
        visual_updates=120, size=16, sheet_size=4):
    if not seeds or pretrain_updates < 3 or visual_updates < 3:
        raise ValueError("Need seeds and at least three updates per stage")
    dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for arm in ARMS:
            rows.append(_arm(seed, arm, dest, pretrain_updates, visual_updates,
                             size, sheet_size))
        # The decisive persistence test: same coherent pretraining, but freeze
        # its connections before visual learning versus continued plasticity.
        rows.append(_arm(seed, "A_coherent_frozen", dest, pretrain_updates,
                         visual_updates, size, sheet_size, freeze_wave=True))
    report = {
        "experiment": "developmental_transfer_v1", "seeds": list(seeds),
        "pretrain_updates": pretrain_updates, "visual_updates": visual_updates,
        "active_3d_updates": visual_updates, "same_posttraining_sequences": True,
        "forecast_scoring": "forecast captured before observation absorption",
        "results": rows,
        "decision_rule": "Prefer an arm only if it improves visual learning speed "
                         "or held-out camera/occlusion metrics across seeds; "
                         "interesting wave coherence alone is not evidence.",
    }
    (dest / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/developmental-transfer")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--pretrain-updates", type=int, default=120)
    p.add_argument("--visual-updates", type=int, default=120)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--sheet-size", type=int, default=4)
    a = p.parse_args()
    print(json.dumps(run(a.out, tuple(a.seeds), a.pretrain_updates,
                         a.visual_updates, a.size, a.sheet_size), indent=2))
