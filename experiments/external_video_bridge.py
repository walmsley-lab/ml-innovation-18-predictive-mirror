"""Framework-neutral paired-history export and frozen external-forecast scoring.

OpenSTL commonly uses [batch,time,channel,height,width]. Export here is an
interchange contract, NOT an OpenSTL integration or trained reference model.
Targets are stored for supervised training/evaluation; a predictor adapter must
receive ONLY the input frames at inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from src import paired_occlusion_histories as world
from src import reappearance_scoring as events


ROOTS = {"train": 11000, "validation": 41000, "test": 71000}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(split, count, context=8, horizon=10, size=32):
    if split not in ROOTS or count < 1 or context < 8 or horizon < 6 or size < 32:
        raise ValueError("Invalid split or paired-history shape")
    return [item for seed in range(ROOTS[split], ROOTS[split] + count)
            for item in world.pair(seed, context, horizon, size)]


def arrays(rows, context, horizon):
    frames = np.stack([r["frames"] for r in rows])
    if frames.shape[1] != context + horizon:
        raise ValueError("Unexpected generated episode length")
    if len(frames) % 2:
        raise ValueError("Paired histories must be even in number")
    if not np.array_equal(frames[::2, context - 1], frames[1::2, context - 1]):
        raise ValueError("Pair final frames differ")
    if np.array_equal(frames[::2, context:], frames[1::2, context:]):
        raise ValueError("Opposing pairs do not have distinct futures")
    # [N,T,H,W,C] -> [N,T,C,H,W]. Keep RGB in float32 [0,1].
    inputs = frames[:, :context].transpose(0, 1, 4, 2, 3).astype(np.float32)
    targets = frames[:, context:context + horizon].transpose(0, 1, 4, 2, 3).astype(np.float32)
    return inputs, targets


def export(out="runs/external-video-bridge", train_pairs=24, validation_pairs=8,
           test_pairs=12, context=8, horizon=10, size=32):
    counts = {"train": train_pairs, "validation": validation_pairs,
              "test": test_pairs}
    if any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError("Each split must have >=1 independent scene pair")
    spans = {name: set(range(ROOTS[name], ROOTS[name]+count))
             for name, count in counts.items()}
    if any(spans[a].intersection(spans[b]) for i,a in enumerate(counts)
           for b in list(counts)[i+1:]):
        raise ValueError("Overlapping train/validation/test seed identities")
    dest = Path(out)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing export")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "dataset": "paired_occlusion_histories",
        "generator": "src.paired_occlusion_histories.pair",
        "axis_order": "N,T,C,H,W", "dtype": "float32",
        "scale": "[0,1]", "context_frames": context,
        "horizon_frames": horizon, "image_size": size,
        "seed_roots": ROOTS, "splits": {},
        "critical_limit": "RGB-only input and supervised target are stored in "
                          "same NPZ for training/evaluation. Pass inputs ONLY "
                          "at inference. Results from these worlds are not "
                          "numerically comparable with Moving-MNIST leaderboards."
    }
    for split, count in counts.items():
        examples = records(split, count, context, horizon, size)
        inputs, targets = arrays(examples, context, horizon)
        path = dest / f"{split}.npz"
        np.savez_compressed(path, inputs=inputs, targets=targets)
        manifest["splits"][split] = {
            "file": path.name, "sha256": sha256(path),
            "scene_pair_ids": list(range(ROOTS[split], ROOTS[split]+count)),
            "independent_scene_pairs": count,
            "episodes": len(examples),
            "inputs_shape": list(inputs.shape),
            "targets_shape": list(targets.shape)
        }
    (dest/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def verify(out):
    folder = Path(out)
    m = json.loads((folder/"manifest.json").read_text())
    if m.get("schema_version") != 1 or m.get("axis_order") != "N,T,C,H,W":
        raise ValueError("Wrong bridge format")
    for name, info in m["splits"].items():
        path = folder / info["file"]
        if not path.is_file() or sha256(path) != info["sha256"]:
            raise ValueError("Missing or modified external dataset: " + name)
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"inputs", "targets"}:
                raise ValueError("Input/target fields changed")
            if list(data["inputs"].shape) != info["inputs_shape"]:
                raise ValueError("Input dimensions changed")
            if list(data["targets"].shape) != info["targets_shape"]:
                raise ValueError("Target dimensions changed")
    return True


def evaluate(out, predictions, split="test"):
    """Score frozen predictions [N,horizon,C,H,W], never model training."""
    folder = Path(out)
    if not verify(folder):
        raise ValueError("Data integrity check failed")
    manifest = json.loads((folder/"manifest.json").read_text())
    if split not in manifest["splits"]:
        raise ValueError("Unknown split")
    info = manifest["splits"][split]
    count = info["independent_scene_pairs"]
    past = manifest["context_frames"]
    horizon = manifest["horizon_frames"]
    size = manifest["image_size"]
    cases = records(split, count, past, horizon, size)
    with np.load(folder / info["file"], allow_pickle=False) as data:
        truth = data["targets"].transpose(0,1,3,4,2)
    with np.load(predictions, allow_pickle=False) as data:
        if set(data.files) != {"predictions"}:
            raise ValueError("Expected only predictions array")
        forecast = np.asarray(data["predictions"],np.float32)
    if forecast.shape != tuple(info["targets_shape"]):
        raise ValueError("Prediction dimensions / order mismatch")
    if not np.isfinite(forecast).all() or np.min(forecast) < 0 or np.max(forecast) > 1:
        raise ValueError("Nonfinite/out-of-range RGB forecasts")
    predicted = forecast.transpose(0,1,3,4,2)
    scored = events.score(predicted, cases, past)
    event_success = []
    for row in scored["results"]:
        per_case = []
        for event in row["events"]:
            position = event["reappearance_centroid_error_px"]
            timing = event["timing_error_frames"]
            per_case.append(int(position is not None and position <= 3
                                and timing is not None and abs(timing) <= 1
                                and event["premature_visible_frames"] == 0))
        if len(per_case) != 1:
            raise ValueError("Expected one return event per episode")
        event_success.append(per_case[0])
    if len(event_success) != count * 2:
        raise ValueError("Expected opposing pairs")
    return {
        "metric_contract": "paired_history_event_v1",
        "split": split, "predictions_sha256": sha256(predictions),
        "dataset_sha256": info["sha256"],
        "rgb_mse": float(np.mean((predicted - truth)**2)),
        "pair_success": np.asarray(event_success).reshape(-1,2).mean(axis=1).tolist(),
        "success_rate": float(np.mean(event_success)),
        "future_contrast_mse": float(np.mean((
            (predicted[::2]-predicted[1::2])-(truth[::2]-truth[1::2]))**2)),
        "event_scores": scored,
        "interpretation": "Scores supplied frozen predictions, not trained-model "
                          "reproduction; provenance of predictor/checkpoint is external."
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/external-video-bridge")
    p.add_argument("--train-pairs", type=int, default=24)
    p.add_argument("--validation-pairs", type=int, default=8)
    p.add_argument("--test-pairs", type=int, default=12)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--score", help="NPZ containing [N,horizon,C,H,W] predictions")
    p.add_argument("--split", choices=("train","validation","test"), default="test")
    a=p.parse_args()
    if a.verify:
        print(json.dumps({"verified": verify(a.out)}))
    elif a.score:
        report = evaluate(a.out,a.score,a.split)
        dest = Path(a.out)/f"{a.split}-external-scores.json"
        dest.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"success_rate":report["success_rate"],
                          "rgb_mse":report["rgb_mse"]},indent=2))
    else:
        m=export(a.out,a.train_pairs,a.validation_pairs,a.test_pairs)
        print(json.dumps({"splits":m["splits"],
                          "axis_order":m["axis_order"]},indent=2))


if __name__ == "__main__":
    main()
