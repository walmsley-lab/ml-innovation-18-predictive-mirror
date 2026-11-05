"""Hash-linked, read-only archive of actual campaign metrics/checkpoints.

Never manufacture a missing artifact or treat a user-supplied commit ID as
verified provenance. The original local campaign directories must be present.
"""
import argparse
import hashlib
import json
import platform
import re
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

SCHEMAS = {
    "perceptual": "paired-diverse-rgb-curriculum-v1",
    "gaussian": "rgb-inferred-gaussian-nca-v1",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_rows(metrics, kind):
    """Every per-seed result has a fixed, independently checkable filename."""
    if kind == "perceptual":
        for entry in metrics["results"]:
            name = f'{entry["model"]}-seed{int(entry["seed"])}-{entry["curriculum"]}-best.npz'
            yield entry, name
    else:
        for entry in metrics["runs"]:
            yield entry, f'seed-{int(entry["seed"])}.npz'


def sample_forecast(kind, checkpoint, entry, metrics, out, width=8, ticks=2):
    """Regenerate one held-out example from the archived selected checkpoint.

    Strictly uses future RGB only as evaluator truth, not as model input.
    """
    context = int(metrics["context"])
    horizon = int(metrics["horizon"])
    size = int(metrics["size"])
    with np.load(checkpoint, allow_pickle=False) as opened:
        loaded = {k: jnp.asarray(opened[k]) for k in opened.files}
    if kind == "perceptual":
        from experiments.rgb_world_ladder_training import specification
        from src import rgb_world_ladder as world
        _, predictor = specification(entry["model"], jax.random.PRNGKey(
            int(entry["seed"])), width, horizon, ticks)
        record = world.split("test", 1, context, horizon, size, 2,
                             "occlusion_long")[0]
        observed = record["frames"]
        predicted = np.asarray(predictor(loaded, jnp.asarray(
            observed[None, :context])))[0]
        source = "test / occlusion_long / seed 91000"
    else:
        from src import world3d, rgb_gaussian_nca as model
        record = world3d.episode(95000, context + horizon, size, "ood_camera")
        observed = record["rgb"]
        params = {k: v for k, v in loaded.items()
                  if not k.startswith("dynamics_")}
        params["dynamics"] = {k.removeprefix("dynamics_"): v
                              for k, v in loaded.items()
                              if k.startswith("dynamics_")}
        predictions = {mode: np.asarray(model.rollout(
            params, jnp.asarray(observed[None, :context]),
            jnp.asarray(record["camera_actions"][None]),
            jnp.asarray(record["K"]), horizon, mode))[0]
            for mode in model.MODES}
        predicted = predictions["cellular"]
        source = "test / ood_camera / seed 95000 / planned actions"
    if predicted.shape != observed[context:context + horizon].shape:
        raise ValueError("Representative forecast shape mismatch")
    payload = {"context_rgb": observed[:context],
               "future_truth_rgb": observed[context:context+horizon],
               "predicted_rgb": predicted}
    if kind == "gaussian":
        payload.update({f"{mode}_rgb": values
                        for mode, values in predictions.items()})
    np.savez_compressed(out, **payload)
    return source


def archive(source, destination, kind, source_commit, width=8, ticks=2,
            examples=True):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if kind not in SCHEMAS:
        raise ValueError("Unknown campaign kind")
    if not re.fullmatch("[a-fA-F0-9]{40}", source_commit):
        raise ValueError("Supply exact 40-character source commit SHA")
    if source == destination or source in destination.parents:
        raise ValueError("Destination cannot overwrite the original campaign")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Archive destination must be empty")
    origin = source / "metrics.json"
    if not origin.is_file():
        raise FileNotFoundError("Missing source metrics.json; cannot archive report")
    metrics = json.loads(origin.read_text())
    if metrics.get("protocol") != SCHEMAS[kind]:
        raise ValueError("Campaign protocol mismatch")
    rows = list(checkpoint_rows(metrics, kind))
    if not rows or len(set(p for _, p in rows)) != len(rows):
        raise ValueError("Missing or duplicate per-seed checkpoint records")
    # Verify complete source and previously logged checkpoint hashes FIRST.
    verified = []
    for entry, filename in rows:
        candidate = source / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing reported checkpoint: {filename}")
        digest = sha256(candidate)
        expected = entry.get("checkpoint_sha256")
        if expected and expected != digest:
            raise ValueError(f"Checkpoint hash disagrees with metrics: {filename}")
        if entry.get("selected_step") is None:
            raise ValueError(f"Missing selected training step: {filename}")
        verified.append((entry, filename, digest))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "checkpoints").mkdir(exist_ok=True)
    (destination / "examples").mkdir(exist_ok=True)
    shutil.copy2(origin, destination / "metrics.json")
    index = []
    used_examples = set()
    for entry, filename, digest in verified:
        copied = destination / "checkpoints" / filename
        shutil.copy2(source / filename, copied)
        if sha256(copied) != digest:
            raise IOError(f"Checkpoint copying changed bytes: {filename}")
        record = {"model": entry.get("model", "rgb_gaussian_nca"),
                  "seed": int(entry["seed"]), "curriculum": entry.get("curriculum"),
                  "selected_step": int(entry["selected_step"]),
                  "checkpoint": f"checkpoints/{filename}",
                  "checkpoint_sha256": digest,
                  "reported_result": ("results" if kind=="perceptual" else "runs"),
                  "sample": None}
        # Representative forecasts: one checkpoint per arm/curriculum, and
        # one seed for the Gaussian campaign; every checkpoint still hashed.
        group = (entry.get("model"), entry.get("curriculum"))
        if examples and group not in used_examples:
            filename_example = filename.removesuffix(".npz") + "-forecast.npz"
            dest = destination / "examples" / filename_example
            record["sample_source"] = sample_forecast(
                kind, copied, entry, metrics, dest, width, ticks)
            record["sample"] = "examples/" + filename_example
            record["sample_sha256"] = sha256(dest)
            used_examples.add(group)
        index.append(record)
    artifact_files = {"metrics.json": sha256(destination / "metrics.json")}
    for row in index:
        artifact_files[row["checkpoint"]] = row["checkpoint_sha256"]
        if row["sample"]:
            artifact_files[row["sample"]] = row["sample_sha256"]
    manifest = {
        "schema": "predictive-mirror-evidence-archive-v1",
        "kind": kind, "source_commit_claimed": source_commit,
        "commit_provenance_verified": False,
        "source_directory": str(source),
        "source_command": "NOT_VERIFIED; record exact original invocation separately",
        "runtime_for_replayed_examples": {
            "python": platform.python_version(), "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": jax.default_backend(),
            "width_assumed": width, "ticks_assumed": ticks,
        },
        "per_seed_results": index, "artifacts": artifact_files,
        "limits": "The metrics and checkpoints are byte-hashed. This tool "
        "does not prove the original run used the declared git revision, "
        "configuration, hardware, or timing procedure. Sample forecasts "
        "are recreated by the installed checkout and require matching "
        "model source; they are not the operator's original outputs.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def verify(folder):
    folder = Path(folder)
    report = json.loads((folder / "manifest.json").read_text())
    if report.get("schema") != "predictive-mirror-evidence-archive-v1":
        raise ValueError("Invalid evidence archive")
    for name, digest in report["artifacts"].items():
        path = folder / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or tampered artifact: {name}")
    return len(report["artifacts"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("archive")
    pack.add_argument("--source", required=True)
    pack.add_argument("--out", required=True)
    pack.add_argument("--kind", choices=SCHEMAS, required=True)
    pack.add_argument("--source-commit", required=True)
    pack.add_argument("--width", type=int, default=8)
    pack.add_argument("--ticks", type=int, default=2)
    pack.add_argument("--no-examples", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("--archive", required=True)
    a = p.parse_args()
    if a.command == "verify":
        print(json.dumps({"checked_artifacts": verify(a.archive)}))
    else:
        m = archive(a.source, a.out, a.kind, a.source_commit,
                    a.width, a.ticks, not a.no_examples)
        print(json.dumps({"rows": len(m["per_seed_results"]),
                          "artifacts": len(m["artifacts"])}))


if __name__ == "__main__":
    main()
