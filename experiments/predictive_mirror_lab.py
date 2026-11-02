"""Offline predictive-mirror laboratory: switch independently saved predictors.

Every model adapter receives ONLY its representation's observed prefix.
The realized future is used afterward for labeled display and scoring.
No model parameters are trained, merged, or altered.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src import predictive_lab_render as renderer

RGB_KINDS = (
    "rgb_nca", "rgb_phase", "rgb_selector", "rgb_multiscale",
    "rgb_soft_references", "rgb_phase_entrainment")
KINDS = RGB_KINDS + ("graph_skeleton", "particle")
DEFAULT_MODE = {
    "rgb_nca": "plain", "rgb_phase": "coupled",
    "rgb_selector": "local", "rgb_multiscale": "multiscale",
    "rgb_soft_references": "learned",
    "rgb_phase_entrainment": "entrained",
    "graph_skeleton": "graph", "particle": "messages"}


def predict_rgb(kind, weights, context, horizon, ticks, mode):
    """Use the native model code for each independent RGB technique."""
    if kind == "rgb_nca":
        from src import predictive_rgb_nca as model
        if mode != "plain":
            raise ValueError("rgb_nca mode must be plain")
        return model.rollout(weights, context, horizon, ticks)
    if kind == "rgb_phase":
        from src import rgb_phase_nca as model
        return model.rollout(weights, context, horizon, ticks, mode)
    if kind == "rgb_selector":
        from src import selectable_rgb_nca as model
        references = model.reference_map(context.shape[2], context.shape[3]) \
            if mode == "reference" else None
        return model.rollout(weights, context, horizon, ticks, mode, references)
    if kind == "rgb_multiscale":
        from src import rgb_multiscale_sheets as model
        return model.rollout(weights, context, horizon, ticks, mode)
    if kind == "rgb_soft_references":
        from src import rgb_soft_references as model
        return model.rollout(weights, context, horizon, ticks, mode)
    if kind == "rgb_phase_entrainment":
        from src import rgb_phase_entrainment as model
        return model.rollout(weights, context, horizon, ticks, mode)
    raise ValueError(f"Not an RGB model: {kind}")


def evaluate(entry, checkpoint, seed, context, horizon, size, ticks):
    """Return independent representation-aware panels and measurements."""
    kind = entry["kind"]
    mode = entry.get("mode", DEFAULT_MODE[kind])
    label = entry["label"]
    with np.load(checkpoint, allow_pickle=False) as archive:
        params = {key: jnp.asarray(archive[key]) for key in archive.files}
    leads = tuple(sorted({horizon, *(h for h in (1, 3, 5, 10)
                                     if h <= horizon)}))
    if kind in RGB_KINDS:
        from src import rgb_benchmark as bench
        samples = bench.frames(bench.scenes(seed, 1, context, horizon, size))
        prefix = samples[:, :context]
        prediction = np.asarray(predict_rgb(
            kind, params, jnp.asarray(prefix), horizon, ticks, mode))[0]
        expected = samples[0, context:context+horizon]
        anchor = samples[0, context-1]
        controls = bench.controls(prefix, horizon)
        baseline = controls["constant_velocity"][0]
        rows = []
        for lead in leads:
            current = prediction[lead-1]
            truth = expected[lead-1]
            model_mse = float(np.square(current-truth).mean())
            base_mse = float(np.square(baseline[lead-1]-truth).mean())
            rows.append({
                "lead": lead,
                "score": f"RGB MSE: model {model_mse:.6f}; velocity {base_mse:.6f}",
                "metrics": {"model_mse": model_mse,
                            "velocity_mse": base_mse},
                "tiles": [
                    ("Observed at issue", anchor),
                    ("Predicted future", current),
                    ("Constant velocity", baseline[lead-1]),
                    ("Actual future", truth),
                    ("Absolute RGB error", np.abs(current-truth))]})
        representation = "RGB video pixels"
        caveat = ("Model forecasts RGB from past observed pixels. "
                  "Future truth is display/scoring only; "
                  "same RGB scene permits side-by-side comparison with RGB techniques.")
    elif kind == "graph_skeleton":
        from src import graph_skeleton_nca as nca
        from experiments import graph_skeleton_nca as world
        from experiments.graph_skeleton_horizons import (
            rollout, last_seen)
        if mode not in ("graph", "no_messages"):
            raise ValueError("graph_skeleton mode must be graph or no_messages")
        # Existing skeleton generator returns T+1 frames for steps=T.
        observed, visibility, truth = world.examples(
            (seed,), steps=context+horizon-1)
        prefix = observed[:, :context]
        masks = visibility[:, :context]
        predicted = np.asarray(rollout(
            params, jnp.asarray(prefix), jnp.asarray(masks), horizon,
            nca.adjacency(), mode == "graph"))[0]
        baseline = last_seen(prefix, masks, horizon, use_velocity=True)[0]
        actual = truth[0, context:context+horizon]
        edges = nca.EDGES
        rows = []
        for lead in leads:
            current = predicted[lead-1]
            target = actual[lead-1]
            model_mae = float(np.linalg.norm(current-target, axis=-1).mean()*32)
            base_mae = float(np.linalg.norm(
                baseline[lead-1]-target, axis=-1).mean()*32)
            rows.append({
                "lead": lead,
                "score": f"2D joint error: model {model_mae:.2f}px; "
                         f"velocity {base_mae:.2f}px (32px coordinate scale)",
                "metrics": {"model_joint_mae_px": model_mae,
                            "velocity_joint_mae_px": base_mae},
                "tiles": [
                    ("Observed joints", renderer.raster_points(
                        prefix[0,-1], edges, visible=masks[0,-1,:,0])),
                    ("Predicted joints", renderer.raster_points(current, edges)),
                    ("Joint velocity", renderer.raster_points(
                        baseline[lead-1], edges)),
                    ("Future joint truth", renderer.raster_points(target, edges))]})
        representation = "Provided 2D skeleton joints and fixed graph"
        caveat = ("Supplied 2D joints and topology, not inferred pose from RGB. "
                  "Joint coordinate error is not comparable to RGB image MSE.")
    elif kind == "particle":
        from src import particle_nca as nca
        from experiments.particle_nca_motion import dataset, controls
        if mode not in ("messages", "no_messages"):
            raise ValueError("particle mode must be messages or no_messages")
        particles = int(entry.get("particles", 2))
        samples = dataset((seed,), context=context, horizon=horizon,
                          size=size, particles=particles)
        prefix = samples[:, :context]
        predicted = np.asarray(nca.rollout(
            params, jnp.asarray(prefix), horizon, mode == "messages"))[0]
        baseline = controls(prefix, horizon)["constant_velocity"][0]
        target = samples[0, context:context+horizon]
        rows = []
        for lead in leads:
            current = predicted[lead-1]
            actual = target[lead-1]
            world_scale = size + 16
            model_mae = float(np.linalg.norm(
                current-actual, axis=-1).mean()*world_scale)
            base_mae = float(np.linalg.norm(
                baseline[lead-1]-actual, axis=-1).mean()*world_scale)
            rows.append({
                "lead": lead,
                "score": f"2D particle error: model {model_mae:.2f}px; "
                         f"velocity {base_mae:.2f}px (world-coordinate scale)",
                "metrics": {"model_particle_mae_px": model_mae,
                            "velocity_particle_mae_px": base_mae},
                "tiles": [
                    ("Observed particles", renderer.raster_points(prefix[0,-1])),
                    ("Predicted particles", renderer.raster_points(current)),
                    ("Particle velocity", renderer.raster_points(baseline[lead-1])),
                    ("Future particle truth", renderer.raster_points(actual))]})
        representation = "Provided tracked 2D particle coordinates"
        caveat = ("Stable particle identities and coordinates are supplied. "
                  "This is not RGB object detection; location error is not image MSE.")
    else:
        raise ValueError(f"Unsupported technique kind: {kind}")

    return {
        "label": label, "kind": kind, "mode": mode,
        "representation": representation, "caveat": caveat,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "rows": rows,
    }


def run(manifest, out="runs/predictive-mirror-lab", seed=71000,
        context=8, horizon=10, size=32, ticks=2, allow_missing=False):
    if context < 2 or horizon < 1 or size < 16 or ticks < 1:
        raise ValueError("Expected context>=2, horizon>=1, size>=16, ticks>=1")
    definition = json.loads(Path(manifest).read_text())
    specs = definition.get("models")
    if not isinstance(specs, list) or not specs:
        raise ValueError("Manifest needs a nonempty models array")
    results, missing = [], []
    labels = set()
    for spec in specs:
        if not isinstance(spec, dict) or any(
                not isinstance(spec.get(k), str) or not spec[k].strip()
                for k in ("label", "kind", "checkpoint")):
            raise ValueError("Every model needs label, kind and checkpoint")
        if spec["kind"] not in KINDS:
            raise ValueError(f"Unknown model kind {spec['kind']}")
        if spec["label"] in labels:
            raise ValueError("Duplicate model label")
        labels.add(spec["label"])
        checkpoint = Path(spec["checkpoint"])
        if not checkpoint.is_file():
            if allow_missing:
                missing.append({"label": spec["label"],
                                "checkpoint": str(checkpoint)})
                continue
            raise FileNotFoundError(checkpoint)
        results.append(evaluate(spec, checkpoint, seed, context, horizon,
                                size, ticks))
    if not results:
        raise ValueError("No available checkpoint models to visualize")
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"index.html").write_text(renderer.html_document(results))
    # Drop raster arrays from JSON; all image pixels are in the self-contained HTML.
    clean = [{
        k: v for k, v in item.items() if k != "rows"}
        | {"per_horizon": [
            {k: v for k, v in row.items() if k != "tiles"}
            for row in item["rows"]]}
        for item in results]
    report = {
        "experiment": "offline_predictive_mirror_laboratory",
        "manifest": str(manifest), "seed": seed, "context": context,
        "horizon": horizon, "size": size, "ticks": ticks,
        "models": clean, "missing_checkpoints": missing,
        "warning": "These models do NOT share input representations: RGB images, "
                   "provided 2D skeletons and tracked particles are different tasks. "
                   "Only compare numerical scores within the same explicit "
                   "representation/benchmark protocol. Use checkpoint training "
                   "records to verify scene seeds are held out; forecasts "
                   "were computed exclusively from the observed prefix."
    }
    (folder/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="runs/predictive-mirror-lab")
    parser.add_argument("--seed", type=int, default=71000)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--allow-missing", action="store_true")
    a = parser.parse_args()
    report = run(a.manifest, a.out, a.seed, a.context, a.horizon,
                 a.size, a.ticks, a.allow_missing)
    print(json.dumps({"models": [m["label"] for m in report["models"]],
                      "missing": report["missing_checkpoints"],
                      "viewer": str(Path(a.out)/"index.html")}, indent=2))


if __name__ == "__main__":
    main()
