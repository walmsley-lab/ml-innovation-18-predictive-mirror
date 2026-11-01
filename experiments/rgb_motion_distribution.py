"""Controlled RGB NCA single-trajectory vs distribution-training experiment.

Separate from all webcam/checkpoint experiments. Every model trains with the
same optimizer, budget and initialization; only training-scene diversity differs.
Validation selects checkpoints; previously unseen *test* scenes score models
and observed-frame persistence / centroid constant-velocity baselines.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca
from train import adamw

MODES = ("single", "distribution")
SCENE_TYPES = ("straight", "bounce", "occlusion")
TRAIN_SEED, VAL_SEED, TEST_SEED = 31000, 41000, 51000


def episode(seed, frames=18, size=32):
    """Render an independent colored moving blob, with optional wall/occluder.

    Returns image frames and a scenario label. No positions or velocities are
    ever given to models or baselines.
    """
    if frames < 3 or size < 16:
        raise ValueError("Need >=3 frames and >=16 pixel resolution")
    rng = np.random.default_rng(int(seed))
    kind = SCENE_TYPES[int(rng.integers(len(SCENE_TYPES)))]
    radius = rng.uniform(1.5, 2.5)
    margin = radius + 1.0
    color = rng.uniform(.55, 1.0, size=3).astype(np.float32)
    color[int(rng.integers(3))] = 1.0
    x, y = rng.uniform(margin + 2, size - margin - 2, size=2)
    angle = rng.uniform(0, 2*np.pi)
    speed = rng.uniform(.75, 1.65)
    vx, vy = speed * np.cos(angle), speed * np.sin(angle)
    if kind == "bounce":
        # Make a real wall reversal likely near the context/forecast boundary.
        side = int(rng.integers(2))
        x = margin + 4 if side == 0 else size - margin - 4
        vx = -abs(speed) if side == 0 else abs(speed)
        vy *= .5
    if kind == "occlusion":
        x = size * .22
        vx = rng.uniform(.8, 1.5)
        vy = rng.uniform(-.35, .35)
        y = rng.uniform(size*.3, size*.7)

    yy, xx = np.mgrid[:size, :size]
    results = []
    for _ in range(frames):
        blob = np.exp(-.5*((xx-x)**2+(yy-y)**2)/(radius**2))
        image = np.full((size, size, 3), .025, np.float32)
        image += .85 * blob[..., None] * color[None, None, :]
        if kind == "occlusion":
            # A fixed visible foreground occluder, not a future-label trick.
            stripe = np.abs(xx - size//2) <= 2
            image[stripe] = .065
        results.append(np.clip(image, 0, 1).astype(np.float32))
        x += vx
        y += vy
        if x < margin:
            x, vx = 2*margin-x, -vx
        elif x > size - margin - 1:
            x, vx = 2*(size-margin-1)-x, -vx
        if y < margin:
            y, vy = 2*margin-y, -vy
        elif y > size - margin - 1:
            y, vy = 2*(size-margin-1)-y, -vy
    return np.stack(results), kind


def scenes(seed_base, count, frames, size):
    if count < 1:
        raise ValueError("Need at least one trajectory")
    records = [episode(seed_base + i, frames, size) for i in range(count)]
    return np.stack([frames for frames, _ in records]), [kind for _, kind in records]


def center(frame):
    """Foreground centroid inferred from RGB pixels, no simulator coordinates."""
    strength = np.maximum(np.max(frame, axis=-1) - .12, 0.0)
    total = strength.sum()
    if total < 0.1:
        return None
    height, width = strength.shape
    yy, xx = np.mgrid[:height, :width]
    return (float((strength * yy).sum() / total),
            float((strength * xx).sum() / total))


def shift(image, dy, dx):
    """Nearest-integer translation with zero padding (never wrap pixels)."""
    height, width = image.shape[:2]
    dy, dx = int(np.rint(dy)), int(np.rint(dx))
    shifted = np.zeros_like(image)
    h = height - abs(dy)
    w = width - abs(dx)
    if h <= 0 or w <= 0:
        return shifted
    source_y = max(0, -dy)
    source_x = max(0, -dx)
    target_y = max(0, dy)
    target_x = max(0, dx)
    shifted[target_y:target_y+h, target_x:target_x+w] = (
        image[source_y:source_y+h, source_x:source_x+w])
    return shifted


def constant_velocity(context, horizon):
    """Observed-centroid velocity and foreground warping, no privileged state.

    Deliberately a strong *image-derived* constant-velocity baseline, not
    an oracle simulator-velocity or dense optical-flow estimator.
    """
    previous, last = context[-2], context[-1]
    prev_center, last_center = center(previous), center(last)
    if prev_center is None or last_center is None:
        return np.repeat(last[None], horizon, axis=0)
    dy, dx = (last_center[0] - prev_center[0],
              last_center[1] - prev_center[1])
    # Approximate a static background from already-observed context only.
    background = np.min(context, axis=0)
    foreground = np.clip(last - background, 0, 1)
    return np.stack([np.clip(background + shift(foreground, dy*(lead+1),
                                                dx*(lead+1)), 0, 1)
                     for lead in range(horizon)])


def baselines(context, horizon):
    """[N,C,H,W,3] -> two [N,horizon,H,W,3] predictors."""
    last = np.repeat(context[:, -1:, ...], horizon, axis=1)
    velocity = np.stack([constant_velocity(frames, horizon) for frames in context])
    return last, velocity


def measure(predicted, examples, context, leads, threshold=.03):
    """Evaluate against future truth; mask uses truth for metrics ONLY."""
    anchors = examples[:, context-1]
    truth = examples[:, context:]
    scores = {}
    for lead in leads:
        target = truth[:, lead-1]
        forecast = predicted[:, lead-1]
        squared = np.square(forecast - target).mean(axis=-1)
        changed = np.abs(target - anchors).mean(axis=-1) > threshold
        location_errors = []
        for predicted_frame, true_frame in zip(forecast, target):
            true_center = center(true_frame)
            if true_center is None:
                continue  # Invisible object: no position ground truth in pixels.
            predicted_center = center(predicted_frame)
            if predicted_center is None:
                # Explicit failed-detection penalty, not a silent skip.
                location_errors.append(float(np.hypot(*true_frame.shape[:2])))
            else:
                location_errors.append(float(np.hypot(
                    predicted_center[0] - true_center[0],
                    predicted_center[1] - true_center[1])))
        scores[str(lead)] = {
            "mse": float(squared.mean()),
            "motion_mse": float(squared[changed].mean()) if changed.any() else None,
            "motion_pixels": int(changed.sum()),
            "visible_location_mae_px": (
                float(np.mean(location_errors)) if location_errors else None),
            "visible_object_frames": len(location_errors),
        }
    return scores


def validation_mse(params, examples, context, horizon, ticks):
    forecast = np.asarray(nca.rollout(params, jnp.asarray(examples[:, :context]),
                                      horizon, ticks))
    return float(np.square(forecast - examples[:, context:context+horizon]).mean())


def run(output="runs/rgb-motion-distribution", seeds=(11, 23, 37), steps=100,
        train_count=96, val_count=18, test_count=24, batch_size=3,
        size=32, context=8, horizon=10, ticks=2, channels=12):
    if (not seeds or steps < 1 or train_count < 2 or val_count < 1
            or test_count < 1 or batch_size < 1 or size < 16 or context < 2
            or horizon < 1 or ticks < 1 or channels < 1):
        raise ValueError("Invalid experiment configuration")

    total = context + horizon
    train, train_types = scenes(TRAIN_SEED, train_count, total, size)
    validation, val_types = scenes(VAL_SEED, val_count, total, size)
    test, test_types = scenes(TEST_SEED, test_count, total, size)
    # No overlapping trajectory identities or seed ranges.
    identity_sets = [set(range(base, base+count)) for base, count in
                     ((TRAIN_SEED, train_count), (VAL_SEED, val_count),
                      (TEST_SEED, test_count))]
    if any(a & b for i, a in enumerate(identity_sets) for b in identity_sets[i+1:]):
        raise ValueError("Train/validation/test trajectory identity overlap")
    leads = tuple(sorted({lead for lead in (1, 3, 5, 10, horizon) if lead <= horizon}))
    persistence, velocity = baselines(test[:, :context], horizon)
    controls = {
        "persistence": measure(persistence, test, context, leads),
        "constant_velocity": measure(velocity, test, context, leads),
    }

    @jax.jit
    def update(params, first, second, batch, step):
        error, gradients = jax.value_and_grad(
            lambda p: nca.loss(p, batch, context, horizon, ticks))(params)
        norm = jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(gradients)))
        gradients = jax.tree.map(lambda g: g / jnp.maximum(1.0, norm), gradients)
        params, first, second = adamw(params, gradients, first, second, step)
        return params, first, second, error

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        initial = nca.init(jax.random.PRNGKey(int(seed)), channels)
        states = {mode: {
            "params": initial, "first": jax.tree.map(jnp.zeros_like, initial),
            "second": jax.tree.map(jnp.zeros_like, initial),
            "best": initial, "best_step": 0,
            "best_val": validation_mse(initial, validation, context, horizon, ticks),
        } for mode in MODES}
        rng = np.random.default_rng(int(seed) + 60000)
        history = []
        for step in range(1, steps+1):
            draw = rng.integers(0, train_count, size=batch_size)
            for mode in MODES:
                batch = (np.repeat(train[:1], batch_size, axis=0)
                         if mode == "single" else train[draw])
                state = states[mode]
                state["params"], state["first"], state["second"], error = update(
                    state["params"], state["first"], state["second"],
                    jnp.asarray(batch), step)
            if step == steps or step % max(1, steps//5) == 0:
                row = {"step": step}
                for mode in MODES:
                    state = states[mode]
                    metric = validation_mse(state["params"], validation,
                                            context, horizon, ticks)
                    row[mode + "_validation_mse"] = metric
                    if metric < state["best_val"]:
                        state["best"], state["best_val"], state["best_step"] = (
                            state["params"], metric, step)
                history.append(row)
                print(f"seed={seed} step={step} "
                      + " ".join(f"{mode}={row[mode+'_validation_mse']:.6f}"
                                 for mode in MODES), flush=True)

        entry = {"seed": int(seed), "arms": {}, "training_history": history}
        for mode in MODES:
            state = states[mode]
            forecasts = np.asarray(nca.rollout(
                state["best"], jnp.asarray(test[:, :context]), horizon, ticks))
            metric = measure(forecasts, test, context, leads)
            entry["arms"][mode] = {
                "best_validation_step": int(state["best_step"]),
                "best_validation_mse": state["best_val"],
                "test": metric,
                "beats_persistence_at_horizon":
                    metric[str(horizon)]["mse"] < controls["persistence"][str(horizon)]["mse"],
                "beats_constant_velocity_at_horizon":
                    metric[str(horizon)]["mse"] < controls["constant_velocity"][str(horizon)]["mse"],
            }
            np.savez_compressed(folder / f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray, state["best"]))
        results.append(entry)

    summary = {}
    for mode in MODES:
        summary[mode] = {
            "mean_horizon_mse": float(np.mean(
                [row["arms"][mode]["test"][str(horizon)]["mse"] for row in results])),
            "beats_persistence_seeds": sum(
                row["arms"][mode]["beats_persistence_at_horizon"] for row in results),
            "beats_constant_velocity_seeds": sum(
                row["arms"][mode]["beats_constant_velocity_at_horizon"] for row in results),
            "step_zero_checkpoints": sum(
                row["arms"][mode]["best_validation_step"] == 0 for row in results),
        }
    report = {
        "experiment": "rgb_nca_single_vs_trajectory_distribution",
        "method": "same RGB NCA architecture, matched parameters and updates; only "
                  "training trajectory diversity varies",
        "seeds": list(map(int, seeds)), "steps_per_arm": steps,
        "train_trajectories": train_count, "validation_trajectories": val_count,
        "test_trajectories": test_count, "size": size, "context": context,
        "horizon": horizon, "ticks": ticks, "channels": channels, "leads": list(leads),
        "split_seed_roots": {"train": TRAIN_SEED, "validation": VAL_SEED, "test": TEST_SEED},
        "scenario_counts": {
            split: {kind: types.count(kind) for kind in SCENE_TYPES}
            for split, types in (("train", train_types), ("validation", val_types),
                                 ("test", test_types))
        },
        "baselines": controls, "results": results, "summary": summary,
        "limitations": "This is a new synthetic benchmark, NOT numerically comparable "
                       "to the earlier uncommitted Step 5 sweep. Test scenarios are unseen "
                       "but share scenario families with training. Constant velocity is "
                       "an observed-image centroid/foreground-shift baseline, not oracle "
                       "simulator velocity or dense optical flow. Validation alone selects "
                       "checkpoint; test is untouched until final evaluation. "
                       "Motion-region masks are scoring-only.",
    }
    (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/rgb-motion-distribution")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-trajectories", type=int, default=96)
    parser.add_argument("--validation-trajectories", type=int, default=18)
    parser.add_argument("--test-trajectories", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--channels", type=int, default=12)
    args = parser.parse_args()
    report = run(args.out, args.seeds, args.steps, args.train_trajectories,
                 args.validation_trajectories, args.test_trajectories,
                 args.batch_size, args.size, args.context, args.horizon,
                 args.ticks, args.channels)
    print(json.dumps({"summary": report["summary"],
                      "baselines_at_horizon": {
                          arm: value[str(args.horizon)]
                          for arm, value in report["baselines"].items()}}, indent=2))


if __name__ == "__main__":
    main()
