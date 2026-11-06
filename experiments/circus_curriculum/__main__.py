"""Run the first Circus Curriculum study: RGB forecasts of rotating 4D objects."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from experiments.accuracy_cost_audit import measure
from experiments.circus_curriculum import model, probes, world
from src import baselines, experiment_harness as harness, rgb_motion_world

CURRICULA = ("ordinary", "hypergeometry", "mixed")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_arrays(folder, filename, **arrays):
    path = folder / filename
    np.savez_compressed(path, **arrays)
    return {"file": filename, "sha256": sha(path)}


def ids_for_split(count, offset):
    if count < 2 or count >= 10_000:
        raise ValueError("Need 2..9999 independent episodes per split")
    return list(range(offset, offset + count))


def ordinary(seeds, steps, size):
    return np.stack([rgb_motion_world.episode(seed, steps, size, world_margin=0)["frames"]
                     for seed in seeds])


def training_pool(physical, geometry, curriculum):
    if physical.shape != geometry.shape or len(physical) % 2:
        raise ValueError("Need matched even-sized training pools")
    if curriculum == "ordinary":
        return physical
    if curriculum == "hypergeometry":
        return geometry
    if curriculum == "mixed":
        # Same total unique clips; half the ordinary exposure is replaced.
        return np.concatenate((physical[:len(physical)//2], geometry[len(geometry)//2:]))
    raise ValueError("Unknown curriculum")


def train(params, clips, validation, schedule, context, horizon):
    predict = jax.jit(lambda p, x: model.rollout(p, x, horizon))

    def loss(p, batch):
        future = batch[:, context:context+horizon]
        changed = jnp.mean(jnp.abs(future - batch[:, context-1:context]), -1, keepdims=True)
        return jnp.mean((1 + 8*changed) * (predict(p, batch[:, :context]) - future)**2)

    update = harness.clipped_adamw_update(loss)
    first, second = harness.init_moments(params)
    val_prefix = jnp.asarray(validation[:, :context])
    val_truth = validation[:, context:context+horizon]

    def validation_score(p):
        value = float(np.mean((np.asarray(predict(p, val_prefix)) - val_truth)**2))
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite validation loss")
        return value

    best, best_step, best_score = params, 0, validation_score(params)
    curve = [{"step": 0, "validation_mse": best_score}]
    for step, indices in enumerate(schedule, 1):
        params, first, second, objective = update(
            params, first, second, (jnp.asarray(clips[indices]),), step)
        if not np.isfinite(float(objective)):
            raise FloatingPointError("Nonfinite training objective")
        if step == 1 or step == len(schedule) or step % max(1, len(schedule)//4) == 0:
            value = validation_score(params)
            curve.append({"step": step, "validation_mse": value})
            if value < best_score:
                best, best_score, best_step = params, value, step
    return best, {"selected_step": best_step, "validation_mse": best_score, "curve": curve}


def score(predictions, clips, context):
    truth = clips[:, context:]
    predictions = np.asarray(predictions)
    if (predictions.shape != truth.shape or not np.isfinite(predictions).all()
            or predictions.min() < 0 or predictions.max() > 1):
        raise ValueError("Invalid prediction shape or RGB values")
    squared = (predictions - truth)**2
    changed = np.mean(np.abs(truth - clips[:, context-1:context]), axis=-1) > .03
    pixel_error = squared.mean(axis=-1)
    return {"rgb_mse": float(squared.mean()),
            "mse_by_lead": squared.mean(axis=(0, 2, 3, 4)).tolist(),
            "episode_mse": squared.mean(axis=(1, 2, 3, 4)).tolist(),
            "changed_pixel_mse": float(pixel_error[changed].mean()) if changed.any() else None,
            "changed_pixels": int(changed.sum())}


def gallery(folder, name, predictions, clips, context):
    from experiments.verified_replay_gallery import panel_arrays, save_panel
    rows = []
    for lead in sorted({1, predictions.shape[1]}):
        path = folder / f"{name}-lead-{lead}.png"
        save_panel(panel_arrays(clips[0, context-1], predictions[0, lead-1],
                                clips[0, context+lead-1]), path, scale=4)
        rows.append({"file": path.name, "sha256": sha(path), "episode": 0,
                     "anchor_frame": context-1, "target_frame": context+lead-1})
    return rows


def latent_probe_report(params, initial, probe_clips, probe_targets, cases,
                        targets, context, folder, key):
    encode = jax.jit(model.encode_history)
    features = {
        "trained_latent": lambda x: np.asarray(encode(params, jnp.asarray(x))),
        "random_latent": lambda x: np.asarray(encode(initial, jnp.asarray(x))),
        "last_image": lambda x: probes.image_features(x, last_only=True),
        "image_history": probes.image_features,
    }
    results = {}
    for name, feature in features.items():
        x = feature(probe_clips[:, :context])
        fitted = probes.fit(x, probe_targets)
        results[name] = {
            "feature_dimension": x.shape[1],
            "checkpoint": save_arrays(folder, f"{key}-probe-{name}.npz", **fitted),
            "cases": {case: probes.score(probes.predict(fitted, feature(clips[:, :context])),
                                         targets[case])
                      for case, clips in cases.items() if case != "ordinary"},
        }
    return {"target": "sin(theta1), sin(theta2), cos(theta1), cos(theta2) at last observation",
            "results": results,
            "limits": "Diagnostic decoding in the simulator's coordinate convention. "
                      "Projection/symmetry ambiguity remains; no claim of a uniquely "
                      "identified 4D state. Probes have different input dimensions. "
                      "No geometry labels train the forecaster or choose its checkpoint."}


def _run(out, seeds, steps, batch_size, train_episodes, validation_episodes,
         test_episodes, probe_episodes, context, horizon, size, latent_dim, render_gallery,
         progress):
    if (not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds)
            or steps < 1 or batch_size < 2 or batch_size % 2 or train_episodes % 2
            or context < 2 or horizon < 1 or size < 16 or size % 8 or latent_dim < 2):
        raise ValueError("Invalid seeds, even training batch/pool, geometry or budget")
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a new or empty output directory")
    if render_gallery:
        from PIL import Image  # noqa: F401
    ids = {name: ids_for_split(count, offset) for name, count, offset in (
        ("train", train_episodes, 0), ("validation", validation_episodes, 10_000),
        ("probe_train", probe_episodes, 20_000), ("test", test_episodes, 30_000))}
    duration = context + horizon
    physical = ordinary(ids["train"], duration, size)
    geometry = world.batch(ids["train"], duration, size)
    validation = np.concatenate((ordinary(ids["validation"], duration, size),
                                 world.batch(ids["validation"], duration, size)))
    folder.mkdir(parents=True, exist_ok=True)
    # Each schedule has the same number of slots from each pool half. Mixed
    # training therefore sees exactly 50% of each domain in every update.
    rng = np.random.default_rng(17)
    half = train_episodes//2
    schedule = np.concatenate((rng.integers(0, half, (steps, batch_size//2)),
                               rng.integers(half, 2*half, (steps, batch_size//2))), axis=1)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    sources = [Path(__file__), Path(world.__file__), Path(model.__file__), Path(probes.__file__),
               root / "src/model.py", root / "src/experiment_harness.py", root / "train.py",
               root / "src/rgb_motion_world.py", root / "src/baselines.py",
               root / "experiments/accuracy_cost_audit.py",
               root / "experiments/verified_replay_gallery.py"]
    protocol = {
        "experiment": "circus-hypergeometry-v1", "model_seeds": list(seeds),
        "split_episode_ids": ids, "steps": steps, "batch_size": batch_size,
        "context": context, "horizon": horizon, "size": size, "latent_dim": latent_dim,
        "curricula": list(CURRICULA), "training_schedule": schedule.tolist(),
        "selection": "lowest RGB MSE on common 50/50 ordinary/hypergeometry validation; "
                     "step zero eligible, earliest tie; never OOD test selection",
        "budget": "same initial weights, updates, total RGB clips and optimizer across curricula; "
                  "mixed replaces half the ordinary exposure, not extra compute",
        "loss": "mean((1+8*mean_channel(abs(future-last)))*squared_error)",
        "optimizer": "shared clipped AdamW at 0.002",
        "source_revision": revision,
        "source_revision_note": "HEAD at launch; file hashes identify uncommitted source too",
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in sources},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "jax": jax.__version__, "device": str(jax.devices("cpu")[0])},
        "cases": {"iid": "new seeds, trained shape/plane/view families",
                  "rotation": "unseen rotation planes, same speed range",
                  "object": "16-cell instead of tesseract",
                  "viewpoint": "camera angle magnitudes [.9,1.2] instead of [0,.3]",
                  "combined": "all three interventions",
                  "ordinary": "new physical moving-disc scenes"},
        "training_data": save_arrays(folder, "training-scenes.npz",
                                      ordinary=physical, hypergeometry=geometry),
        "validation_data": save_arrays(folder, "validation-scenes.npz", frames=validation),
    }
    write_json(folder / "protocol.json", protocol)
    selected = {}
    for seed in seeds:
        initial = model.init(jax.random.PRNGKey(seed), latent_dim)
        for curriculum in CURRICULA:
            key = f"{curriculum}-seed-{seed}"
            if progress:
                print(f"Training {key}", flush=True)
            clips = training_pool(physical, geometry, curriculum)
            params, choice = train(initial, clips, validation, schedule, context, horizon)
            choice["checkpoint"] = save_arrays(folder, f"{key}.npz", **jax.tree.map(np.asarray, params))
            selected[key] = {"seed": seed, "curriculum": curriculum, **choice}
        jax.clear_caches()
    write_json(folder / "selection.json", selected)
    # OOD/ordinary test frames are generated only after all selections freeze.
    cases = {case: world.batch(ids["test"], duration, size, case) for case in world.CASES}
    cases["ordinary"] = ordinary(ids["test"], duration, size)
    test_artifacts = {name: save_arrays(folder, f"test-{name}.npz", frames=clips)
                      for name, clips in cases.items()}
    probe_clips = world.batch(ids["probe_train"], context, size)
    probe_targets = world.phase_targets(ids["probe_train"], context, size)
    targets = {case: world.phase_targets(ids["test"], context, size, case) for case in world.CASES}
    references = {}
    for name, function in (("persistence", baselines.persistence_forecast),
                           ("constant_velocity", baselines.constant_velocity_forecast)):
        references[name] = {}
        for case, clips in cases.items():
            predicted = np.stack([function(c[:context], horizon) for c in clips])
            artifact = save_arrays(folder, f"{name}-{case}.npz", predictions=predicted)
            references[name][case] = {**score(predicted, clips, context), "predictions": artifact}
    for curriculum in CURRICULA:
        key = f"history_ridge-{curriculum}"
        fitted = probes.fit_history_forecast(training_pool(physical, geometry, curriculum), context, horizon)
        save_arrays(folder, f"{key}-weights.npz", **fitted)
        references[key] = {}
        for case, clips in cases.items():
            predicted = probes.history_forecast(fitted, clips[:, :context], horizon)
            artifact = save_arrays(folder, f"{key}-{case}.npz", predictions=predicted)
            references[key][case] = {**score(predicted, clips, context), "predictions": artifact}
    results = {}
    for key, choice in selected.items():
        if progress:
            print(f"Evaluating frozen {key} and its latent probes", flush=True)
        params, digest = harness.load_checkpoint(folder / choice["checkpoint"]["file"])
        if digest != choice["checkpoint"]["sha256"]:
            raise ValueError("Frozen checkpoint hash mismatch")
        predict = jax.jit(lambda p, x: model.rollout(p, x, horizon))
        scores = {}
        for case, clips in cases.items():
            predicted = np.asarray(predict(params, jnp.asarray(clips[:, :context])))
            artifact = save_arrays(folder, f"{key}-{case}.npz", predictions=predicted)
            scores[case] = {**score(predicted, clips, context), "predictions": artifact}
            if render_gallery:
                scores[case]["gallery"] = gallery(folder, f"{key}-{case}", predicted, clips, context)
        prefix = cases["iid"][:1, :context]
        timing = measure(lambda: np.asarray(predict(params, jnp.asarray(prefix))), repeats=5)
        initial = model.init(jax.random.PRNGKey(choice["seed"]), latent_dim)
        latent = latent_probe_report(params, initial, probe_clips, probe_targets,
                                     cases, targets, context, folder, key)
        reset = np.repeat(cases["iid"][:, context-1:context], context, axis=1)
        results[key] = {"selection": choice, "cases": scores, "latent_probes": latent,
                        "history_reset_iid": score(np.asarray(predict(params, jnp.asarray(reset))),
                                                   cases["iid"], context),
                        "timing": timing, "latent_state_bytes": latent_dim * 4}
        jax.clear_caches()
    transfer = []
    for seed in seeds:
        base = np.asarray(results[f"ordinary-seed-{seed}"]["cases"]["ordinary"]["episode_mse"])
        mixed = np.asarray(results[f"mixed-seed-{seed}"]["cases"]["ordinary"]["episode_mse"])
        transfer.append({"seed": seed, "mixed_minus_ordinary_mse": float(np.mean(mixed-base)),
                         "paired_episode_differences": (mixed-base).tolist()})
    report = {"status": "exploratory synthetic curriculum pilot; no automatic success claim",
              "protocol_sha256": sha(folder / "protocol.json"),
              "selection_sha256": sha(folder / "selection.json"), "test_artifacts": test_artifacts,
              "results": results, "references": references, "ordinary_transfer": transfer,
              "limits": ["Latent dimension is representational, not inferred physical dimension.",
                         "Phase probes are coordinate-dependent diagnostics, not identification of 4D structure.",
                         "Deterministic model: no learned posterior, calibrated uncertainty or multiple futures.",
                         "History-ridge references use a closed-form fit on the same clip pools; "
                         "their compute and capacity are not matched to SGD.",
                         "Timing is warmed-up CPU context-to-horizon for one IID clip, excluding capture; "
                         "latent_state_bytes excludes RGB buffers and all other memory.",
                         "Worlds B/C/D, action conditioning and cross-modal transfer are not implemented here.",
                         "Small seed/episode counts and update budgets cannot establish generalization superiority."]}
    write_json(folder / "metrics.json", report)
    return report


def run(out="runs/circus-hypergeometry", seeds=(11,), steps=40, batch_size=4,
        train_episodes=32, validation_episodes=8, test_episodes=8, probe_episodes=32,
        context=8, horizon=6, size=32, latent_dim=64, render_gallery=False, progress=False):
    with jax.default_device(jax.devices("cpu")[0]):
        return _run(out, seeds, steps, batch_size, train_episodes, validation_episodes,
                    test_episodes, probe_episodes, context, horizon, size, latent_dim,
                    render_gallery, progress)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/circus-hypergeometry")
    p.add_argument("--seeds", type=int, nargs="+", default=[11])
    for name, default in (("steps", 40), ("batch-size", 4), ("train-episodes", 32),
                          ("validation-episodes", 8), ("test-episodes", 8),
                          ("probe-episodes", 32), ("context", 8), ("horizon", 6),
                          ("size", 32), ("latent-dim", 64)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--gallery", dest="render_gallery", action="store_true")
    args = p.parse_args()
    report = run(**vars(args), progress=True)
    print(json.dumps(report["ordinary_transfer"], indent=2))
    print(f"Report: {Path(args.out) / 'metrics.json'}")


if __name__ == "__main__":
    main()
