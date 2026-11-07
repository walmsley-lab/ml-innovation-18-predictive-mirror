"""Matched CPU pilot: compact predictive mirror, learned references and controls.

One command trains on the same RGB batches, freezes validation-selected weights,
then saves forecasts, scores and host-to-host timings on disjoint scenes.
Synthetic frames are not wall-clock seconds; this is not a webcam benchmark.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import platform
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from experiments.accuracy_cost_audit import measure
from experiments.paired_frozen_comparison import summarize as paired_score
from src import baselines, compact_predictive_mirror as compact
from src import convlstm_reference as convlstm, experiment_harness as harness
from src import history_baselines, paired_occlusion_histories as paired
from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark
from src import rgb_motion_transport as transport, rgb_motion_world as world
from src.accuracy_cost_frontier import nondominated

MODELS = ("compact", "full_resolution", "nca", "convlstm", "transport")
CONTROLS = {
    "persistence": baselines.persistence_forecast,
    "constant_velocity": baselines.constant_velocity_forecast,
    "history_linear": history_baselines.history_linear,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def split_ids(steps, batch_scenes, validation_scenes, test_scenes):
    counts = (steps * batch_scenes, validation_scenes, test_scenes)
    if min(counts) < 1 or max(counts) >= 1_000_000:
        raise ValueError("Need 1..999999 scene IDs per split")
    return {name: list(range(start, start + count))
            for name, start, count in zip(
                ("train", "validation", "test"),
                (0, 1_000_000, 2_000_000), counts)}


def scenes(ids, context=8, horizon=10, size=32):
    """One future wall bounce and one opposing occlusion pair per scene ID.

    Rejection sampling uses generator truth to construct the bounce task only.
    The returned frame tensor is the sole training/inference input.
    """
    if context < 8 or horizon < 6 or size < 32 or not len(ids):
        raise ValueError("Need scene IDs, context>=8, horizon>=6, size>=32")
    records, frames, bounce_seeds = [], [], []
    for scene in ids:
        for attempt in range(128):
            seed = int(scene) * 128 + attempt
            bounce = world.episode(seed, context + horizon, size, world_margin=0)
            turns = np.any(np.diff(bounce["velocities"], axis=0)[context-1:] != 0)
            if turns:
                break
        else:
            raise ValueError(f"No future wall bounce for scene {scene}")
        pair = paired.pair(int(scene), context, horizon, size)
        # The paired world has only a few motion templates. Vary appearance
        # per scene, identically for both histories, to avoid byte-identical
        # clips across seed splits. This remains a shared synthetic family.
        rng = np.random.default_rng(int(scene))
        gain = rng.uniform(.8, 1., 3).astype(np.float32)
        bias = rng.uniform(0., .015, 3).astype(np.float32)
        bounce["frames"] = bounce["frames"] * gain + bias
        for record in pair:
            record["frames"] = record["frames"] * gain + bias
            record["color"] = record["color"] * gain + bias
        frames.extend((bounce["frames"], pair[0]["frames"], pair[1]["frames"]))
        records.extend(pair)
        bounce_seeds.append(seed)
    return np.stack(frames), records, bounce_seeds


def predictor(name, horizon, ticks):
    if name in ("compact", "full_resolution"):
        factor = 2 if name == "compact" else 1
        return lambda p, x: compact.rollout(p, x, horizon, ticks, factor)
    if name == "nca":
        return lambda p, x: nca.rollout(p, x, horizon, ticks)
    if name == "convlstm":
        return lambda p, x: convlstm.rollout(p, x, horizon)
    if name == "transport":
        return lambda p, x: transport.rollout(p, x, horizon, mode="transport")
    raise ValueError("Unknown predictor")


def initialize(name, seed, width):
    key = jax.random.PRNGKey(seed)
    if name in ("compact", "full_resolution"):
        return compact.init(key, width)
    return {"nca": nca.init, "convlstm": convlstm.init,
            "transport": transport.init}[name](key, width)


def objective(params, clips, predict, context, horizon):
    """Identical RGB-derived motion weighting for every learned model."""
    future = clips[:, context:context+horizon]
    changed = jnp.mean(jnp.abs(future - clips[:, context-1:context]),
                       axis=-1, keepdims=True)
    return jnp.mean((1. + 8. * changed) * (predict(params, clips[:, :context]) - future)**2)


def fit(params, predict, update, train_ids, validation, steps, batch_scenes,
        context, horizon, size):
    """No test data enters checkpoint selection; initialization is eligible."""
    first, second = harness.init_moments(params)
    prefix = jnp.asarray(validation[:, :context])
    truth = validation[:, context:context+horizon]

    def validation_mse(p):
        value = float(np.mean((np.asarray(predict(p, prefix)) - truth)**2))
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite validation score")
        return value

    best, best_step, best_score = params, 0, validation_mse(params)
    curve = [{"step": 0, "validation_mse": best_score}]
    for step in range(1, steps + 1):
        ids = train_ids[(step-1)*batch_scenes:step*batch_scenes]
        batch, _, _ = scenes(ids, context, horizon, size)
        params, first, second, loss = update(
            params, first, second, (jnp.asarray(batch),), step)
        if not np.isfinite(float(loss)):
            raise FloatingPointError("Nonfinite training objective")
        if step == 1 or step == steps or step % max(1, steps // 4) == 0:
            score = validation_mse(params)
            curve.append({"step": step, "validation_mse": score,
                          "training_objective": float(loss)})
            if score < best_score:
                best, best_step, best_score = params, step, score
    return best, {"selected_step": best_step, "validation_mse": best_score,
                  "curve": curve}


def score(predicted, clips, records, context):
    truth = clips[:, context:]
    predicted = np.asarray(predicted)
    if (predicted.shape != truth.shape or not np.isfinite(predicted).all()
            or predicted.min() < 0 or predicted.max() > 1):
        raise ValueError("Invalid forecast shape, finiteness or RGB range")
    pair_indices = np.arange(len(clips)) % 3 != 0
    paired_metrics = paired_score(predicted[pair_indices], truth[pair_indices],
                                  records, context)
    mse = ((predicted - truth)**2).mean(axis=(1, 2, 3, 4))
    return {
        "rgb_mse": float(mse.mean()),
        "scene_mse": mse.reshape(-1, 3).mean(axis=1).tolist(),
        "bounce": benchmark.score(predicted[::3], clips[::3], context,
                                  harness.leads(truth.shape[1])),
        "occlusion": paired_metrics,
    }


def payload_bytes(tree):
    return sum(int(x.size * x.dtype.itemsize) for x in jax.tree.leaves(tree))


def state_bytes(name, params, prefix):
    frame = jnp.asarray(prefix[:1, -1])
    if name in ("compact", "full_resolution"):
        state = compact.initial_state(params, frame, 2 if name == "compact" else 1)
    elif name == "nca":
        state = nca.initial_state(params, 1, frame.shape[1], frame.shape[2])
    elif name == "convlstm":
        state = (*convlstm.initial(params, 1, frame.shape[1], frame.shape[2]), frame)
    else:
        state = (frame, frame)
    return payload_bytes(state)


def timed_forecasts(forecast, prefix, repeats):
    """Batch-one context-to-horizon latency on every test clip, including copies.

    NumPy materialization synchronizes JAX and includes returned RGB transfer.
    Compilation is excluded by warmup. Capture, scoring and PNGs are excluded.
    """
    predictions, measurements = [], []
    for clip in prefix:
        invoke = lambda: np.asarray(forecast(clip[None]))
        timing = measure(invoke, warmup=1, repeats=repeats)
        measurements.extend(timing["samples_ms"])
        predictions.append(invoke()[0])
    return np.stack(predictions), {
        "latency_ms": float(np.median(measurements)),
        "p90_latency_ms": float(np.quantile(measurements, .9)),
        "samples_ms": measurements, "repeats_per_clip": repeats,
        "timed_clips": len(prefix), "batch_size": 1,
    }


def artifact(folder, filename, **arrays):
    path = folder / filename
    np.savez_compressed(path, **arrays)
    return {"file": filename, "sha256": sha(path)}


def gallery(folder, name, predicted, clips, context):
    from experiments.verified_replay_gallery import panel_arrays, save_panel
    entries = []
    for episode, kind in enumerate(("bounce", "occlusion_right", "occlusion_left")):
        for lead in harness.leads(predicted.shape[1], (1, 5, 10)):
            filename = f"{name}-{kind}-lead-{lead}.png"
            # The target index is context+lead-1, never context+lead.
            save_panel(panel_arrays(clips[episode, context-1],
                       predicted[episode, lead-1], clips[episode, context+lead-1]),
                       folder / filename, scale=4)
            entries.append({"file": filename, "sha256": sha(folder / filename),
                            "episode": episode, "kind": kind, "lead": lead,
                            "anchor_frame": context-1,
                            "target_frame": context+lead-1})
    return entries


def save_report_page(folder, report):
    rows, pictures = [], []
    for name, row in report["results"].items():
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{row['rgb_mse']:.6f}</td>"
                    f"<td>{row['occlusion']['success_rate']:.3f}</td>"
                    f"<td>{row['cost']['latency_ms']:.3f}</td>"
                    f"<td>{row['cost']['p90_latency_ms']:.3f}</td></tr>")
        for item in row.get("gallery", []):
            title = f"{name}: {item['kind']}, lead +{item['lead']}"
            pictures.append(f"<figure><figcaption>{html.escape(title)}</figcaption>"
                            f"<img loading='lazy' src='{item['file']}' "
                            f"alt='{html.escape(title)}'></figure>")
    (folder / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Predictive mirror comparison</title><style>"
        "body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem}"
        "td,th{padding:.5rem;text-align:left;border-bottom:1px solid #ccc}"
        "img{max-width:100%}figure{margin:2rem 0}</style>"
        "<h1>Predictive mirror comparison</h1>"
        "<p>Synthetic held-out pilot. Lower MSE and latency are better; higher "
        "reappearance success is better. Forecast horizons are frames, not seconds. "
        "Latency covers one RGB context to its entire open-loop horizon on CPU.</p>"
        "<p><a href='metrics.json'>Full metrics and limitations</a> · "
        "<a href='protocol.json'>Frozen protocol</a></p>"
        "<table><thead><tr><th>Model / seed</th><th>RGB MSE</th>"
        "<th>Reappearance success</th><th>Median ms</th><th>P90 ms</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p>Panels: last observation, issued forecast, actual future, absolute error "
        "(fixed 0–1 scale). Offline pairing; no live timing claim.</p>"
        + "".join(pictures) + "</html>")


def _run(out, seeds, models, steps, batch_scenes, validation_scenes, test_scenes,
         size, context, horizon, width, ticks, repeats, render_gallery, progress):
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(s < 0 or s >= 2**32 for s in seeds)
            or not models or len(set(models)) != len(models)
            or any(m not in MODELS for m in models)
            or min(steps, batch_scenes, validation_scenes, test_scenes, ticks) < 1
            or width < 2 or repeats < 3 or size % 2):
        raise ValueError("Invalid model, seed, geometry or training/timing budget")
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Output directory must be empty; preserve previous evidence")
    if render_gallery:
        # Fail before training if the optional renderer isn't installed.
        from PIL import Image  # noqa: F401
    ids = split_ids(steps, batch_scenes, validation_scenes, test_scenes)
    validation, _, _ = scenes(ids["validation"], context, horizon, size)
    test, test_records, bounce_seeds = scenes(ids["test"], context, horizon, size)
    folder.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    source_files = [Path(__file__), Path(compact.__file__), Path(nca.__file__),
                    Path(transport.__file__), Path(convlstm.__file__),
                    Path(world.__file__), Path(paired.__file__),
                    Path(baselines.__file__), Path(history_baselines.__file__),
                    Path(harness.__file__), Path(benchmark.__file__)]
    source_files.extend(repository / path for path in (
        "src/model.py", "train.py", "src/reappearance_scoring.py",
        "src/rgb_event_metrics.py", "src/accuracy_cost_frontier.py",
        "experiments/paired_frozen_comparison.py", "experiments/accuracy_cost_audit.py",
        "experiments/verified_replay_gallery.py"))
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip()
                    for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    protocol = {
        "protocol": "compact-mirror-pilot-v1", "source_revision": revision,
        "source_sha256": {str(p.relative_to(repository)): sha(p) for p in source_files},
        "split_scene_ids": ids, "test_bounce_generator_seeds": bounce_seeds,
        "test_data": artifact(folder, "test-scenes.npz", frames=test),
        "validation_data": artifact(folder, "validation-scenes.npz", frames=validation),
        "model_seeds": list(seeds), "models": list(models), "steps": steps,
        "batch_scenes": batch_scenes, "clips_per_scene": 3,
        "context": context, "horizon_frames": horizon, "size": size,
        "width": width, "ticks": ticks, "repeats": repeats,
        "selection": "lowest validation RGB MSE, including step zero; earliest tie",
        "training_objective": "mean((1+8*mean_channel(abs(future-last)))*squared_error)",
        "optimizer": "shared clipped AdamW, rate 0.002, same updates and RGB batches",
        "selection_weighting": "one bounce and two occlusion clips per scene, equal clip weight",
        "input": "past RGB only; simulator metadata used only for construction/scoring",
        "budget": "matched updates/data/width; not matched parameters, FLOPs or training time",
        "scene_family": "wall bounces plus the existing discrete paired-history "
                        "motion templates; per-scene RGB gain/bias varies appearance, "
                        "not held-out physical rules",
        "controls": "history_linear assumes one saturated blob and static background; "
                    "constant_velocity estimates global image translation from two frames",
        "device": str(jax.devices("cpu")[0]), "platform": platform.platform(),
        "cpu": cpu, "python_version": platform.python_version(),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
    }
    write_json(folder / "protocol.json", protocol)
    protocol_hash = sha(folder / "protocol.json")
    selections = {}
    for name in models:
        predict = jax.jit(predictor(name, horizon, ticks))
        update = harness.clipped_adamw_update(
            lambda p, x: objective(p, x, predict, context, horizon))
        for seed in seeds:
            key = f"{name}-seed-{seed}"
            if progress:
                print(f"Training {key} on shared RGB batches", flush=True)
            params, selected = fit(initialize(name, seed, width), predict, update,
                                  ids["train"], validation, steps, batch_scenes,
                                  context, horizon, size)
            selected["checkpoint"] = artifact(folder, f"{key}-weights.npz",
                                               **jax.tree.map(np.asarray, params))
            selections[key] = {"model": name, "seed": seed, **selected}
        # Different models compile different training graphs; release executables.
        jax.clear_caches()
    write_json(folder / "selection.json", selections)
    selection_hash = sha(folder / "selection.json")
    results = {}
    prefix = test[:, :context].copy()

    def evaluate(key, forecast, params=None, name=None):
        if progress:
            print(f"Scoring and timing {key} on untouched test scenes", flush=True)
        predictions, cost = timed_forecasts(forecast, prefix, repeats)
        cost.update({"parameter_bytes": 0 if params is None else payload_bytes(params),
                     "state_payload_bytes": None if params is None else
                         state_bytes(name, params, prefix),
                     "peak_memory_mib": None, "energy_j": None})
        committed = artifact(folder, f"{key}-predictions.npz", predictions=predictions)
        row = {**score(predictions, test, test_records, context), "cost": cost,
               "predictions": committed}
        if render_gallery:
            row["gallery"] = gallery(folder, key, predictions, test, context)
        results[key] = row

    for name, function in CONTROLS.items():
        evaluate(name, lambda x, fn=function: np.stack([fn(c, horizon) for c in x]))
    for key, selected in selections.items():
        name = selected["model"]
        params, digest = harness.load_checkpoint(folder / selected["checkpoint"]["file"])
        if digest != selected["checkpoint"]["sha256"]:
            raise ValueError("Selected checkpoint changed before evaluation")
        predict = jax.jit(predictor(name, horizon, ticks))
        evaluate(key, lambda x: predict(params, jnp.asarray(x)), params, name)
        results[key]["selection"] = selected
        if name == "compact":
            reset = np.repeat(prefix[:, -1:], context, axis=1)
            no_feedback = jax.jit(lambda p, x: compact.rollout(
                p, x, horizon, ticks, 2, feedback=False))
            results[key]["diagnostics"] = {
                "history_reset": score(np.asarray(predict(params, jnp.asarray(reset))),
                                       test, test_records, context),
                "error_feedback_disabled": score(
                    np.asarray(no_feedback(params, jnp.asarray(prefix))),
                    test, test_records, context),
                "interpretation": "frozen checkpoint sensitivity, not separately trained "
                                  "ablations or proof of beneficial memory",
            }
        jax.clear_caches()
    rows = [{"model": name, "protocol": protocol_hash, "device": protocol["device"],
             "negative_rgb_mse": -row["rgb_mse"],
             "success_rate": row["occlusion"]["success_rate"],
             "latency_ms": row["cost"]["latency_ms"]}
            for name, row in results.items()]
    report = {
        "protocol_file": "protocol.json", "protocol_sha256": protocol_hash,
        "selection_file": "selection.json", "selection_sha256": selection_hash,
        "status": "synthetic held-out pilot; no general performance claim",
        "results": results,
        "mse_latency_frontier": nondominated(rows, success_key="negative_rgb_mse"),
        "reappearance_latency_frontier": nondominated(rows),
        "cost_scope": "CPU batch-one past-RGB context to complete future horizon; "
                      "host input/output conversion included, compilation warmed up; "
                      "capture, scoring and rendering excluded",
        "memory_scope": "state_payload_bytes counts recurrent carry arrays including "
                        "RGB buffers, excluding input context, outputs, temporaries and "
                        "allocator overhead; it is not measured peak memory",
        "limitations": ["Equal updates and width do not establish baseline training sufficiency.",
                        "Scene pairs are correlated; raw per-scene metrics are retained, "
                        "and seed results are not pooled as independent episodes.",
                        "The frontier describes this pilot only; no significance claim.",
                        "Reappearance success is not identity preservation or webcam transfer.",
                        "Compact recurrence still uses dense RGB transport; no sparse-compute claim.",
                        "Error maps are observed discrepancies, not calibrated uncertainty."],
    }
    write_json(folder / "metrics.json", report)
    save_report_page(folder, report)
    return report


def run(out="runs/compact-mirror", seeds=(11,), models=MODELS, steps=50,
        batch_scenes=1, validation_scenes=2, test_scenes=3, size=32,
        context=8, horizon=10, width=8, ticks=2, repeats=5,
        render_gallery=False, progress=False):
    # Classical NumPy and learned references share the CPU for this first study.
    with jax.default_device(jax.devices("cpu")[0]):
        return _run(out, seeds, models, steps, batch_scenes, validation_scenes,
                    test_scenes, size, context, horizon, width, ticks, repeats,
                    render_gallery, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/compact-mirror")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    for name, default in (("steps", 50), ("batch-scenes", 1), ("validation-scenes", 2),
                          ("test-scenes", 3), ("size", 32), ("context", 8),
                          ("horizon", 10), ("width", 8), ("ticks", 2), ("repeats", 5)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--gallery", action="store_true", dest="render_gallery")
    args = parser.parse_args()
    report = run(**vars(args), progress=True)
    print(json.dumps({name: {"rgb_mse": row["rgb_mse"],
                     "latency_ms": row["cost"]["latency_ms"]}
                     for name, row in report["results"].items()}, indent=2))
    print(f"Report: {Path(args.out) / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
