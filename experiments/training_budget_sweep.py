"""Train several model families at increasing step budgets and track whether
each one is undertrained, collapsed, or genuinely learning.

Arms
----
Five independent families on identical worlds -- ``cnn``, ``transport``,
``nca`` (the collapsing baseline, kept as a control), ``routing`` (learned soft
references) and ``hierarchical`` (two-scale surprise) -- plus three anti-collapse
NCA treatments that share the baseline's cellular core:

* ``nca_fg`` -- foreground-sensitive loss;
* ``nca_blend`` -- non-saturating convex-blend output head;
* ``nca_grounded`` -- blend head + foreground loss + balanced scene sampling.

For each seed an arm is trained once to the largest budget and a checkpoint is
saved at every budget in ``--budgets`` (default 300/600/1200). At each budget the
validation set is scored on **foreground and localization**, not only MSE, using
the event metrics and the collapse diagnostics, and a per-budget status label is
derived so undertraining, collapse and learning are distinguishable.

The predictor only ever sees RGB context frames; palette/visibility truth is
read by the evaluator after each forecast is committed.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import (predictive_rgb_nca as nca, rgb_grounded_nca as grounded,
                 rgb_motion_transport as transport, tiny_cnn_video as cnn,
                 rgb_soft_references as routing,
                 rgb_hierarchical_surprise as hierarchical,
                 rgb_benchmark as bench, rgb_motion_world, rgb_event_metrics,
                 forecast_diagnostics)
from train import adamw


def arm_specification(name, key, width, context, horizon, ticks):
    """Return (params, loss_fn(params,batch), rollout_fn(params,context), balanced)."""
    if name == "cnn":
        return (cnn.init(key, width),
                lambda p, x: cnn.loss(p, x, context, horizon),
                lambda p, c: cnn.rollout(p, c, horizon), False)
    if name == "transport":
        return (transport.init(key, width),
                lambda p, x: transport.loss(p, x, context, horizon, "combined"),
                lambda p, c: transport.rollout(p, c, horizon, "combined"), False)
    if name == "routing":
        return (routing.init(key, width),
                lambda p, x: routing.loss(p, x, context, horizon, ticks, "learned"),
                lambda p, c: routing.rollout(p, c, horizon, ticks, "learned"), False)
    if name == "hierarchical":
        return (hierarchical.init(key, width),
                lambda p, x: hierarchical.loss(p, x, context, horizon, "two_scale_rgb"),
                lambda p, c: hierarchical.rollout(p, c, horizon, "two_scale_rgb"), False)
    # NCA control and anti-collapse treatments.
    nca_arms = {
        "nca": ("residual", 0.0, False),
        "nca_fg": ("residual", 6.0, False),
        "nca_blend": ("blend", 0.0, False),
        "nca_grounded": ("blend", 6.0, True),
    }
    if name in nca_arms:
        output, fg_weight, balanced = nca_arms[name]
        params = grounded.init(key, width, output)
        return (params,
                lambda p, x: grounded.loss(p, x, context, horizon, ticks,
                                           output, fg_weight),
                lambda p, c: grounded.rollout(p, c, horizon, ticks, output),
                balanced)
    raise ValueError(f"Unknown arm {name!r}")


ARMS = ("cnn", "transport", "nca", "routing", "hierarchical",
        "nca_fg", "nca_blend", "nca_grounded")


def foreground_episodes(records, context):
    """Indices of episodes whose FUTURE frames actually contain foreground."""
    flags = []
    for i, record in enumerate(records):
        future = record["frames"][context:]
        flags.append(bool((future.max(axis=-1) > grounded.FOREGROUND_LEVEL).any()))
    present = [i for i, f in enumerate(flags) if f]
    return present if present else list(range(len(records)))


def validation_report(rollout_fn, params, records, frames, context, horizon):
    """Foreground + localization + collapse diagnostics, not just MSE."""
    leads = tuple(sorted({horizon, *(k for k in (1, 3, 5, 10) if k <= horizon)}))
    forecast = np.asarray(rollout_fn(params, jnp.asarray(frames[:, :context])))
    truth = frames[:, context:context + horizon]
    events = rgb_event_metrics.score(forecast, records, context, leads)
    collapse = forecast_diagnostics.collapse_report(forecast, leads)
    final = str(horizon)
    row = events[final]
    return {
        "mse": float(np.square(forecast - truth).mean()),
        "foreground_mse": row["foreground_mse"],
        "foreground_recall": row["foreground_recall"],
        "foreground_iou": row["foreground_iou"],
        "soft_centroid_mae_px": row["soft_centroid_mae_px"],
        "predicted_mass_fraction": row["predicted_mass_fraction"],
        "missing_visible_predictions": row["missing_visible_predictions"],
        "visible_objects": row["visible_objects"],
        "clip_rail_saturation": collapse["clip_rail_saturation"][final],
        "input_independence": collapse["input_independence"][final],
    }


def classify(report):
    """undertrained | collapsed | learning, from the validation report."""
    saturation = report["clip_rail_saturation"]
    independence = report["input_independence"]
    mass = report["predicted_mass_fraction"] or 0.0
    recall = report["foreground_recall"] or 0.0
    if saturation >= 0.9 and independence <= 1e-3 and mass < 0.05:
        return "collapsed"
    if mass >= 0.3 and recall >= 0.2:
        return "learning"
    return "undertrained"


def run(out="runs/training-budget-sweep", arms=ARMS, seeds=(11, 23, 41),
        budgets=(300, 600, 1200), context=8, horizon=10, size=32, width=8,
        ticks=2, batch_size=2, train_count=48, val_count=12):
    budgets = tuple(sorted(set(int(b) for b in budgets)))
    if (not arms or not seeds or not budgets or min(budgets) < 1
            or context < 2 or horizon < 1 or width < 1 or batch_size < 1):
        raise ValueError("Invalid sweep configuration")
    world = {"balls": 2, "occlusion": True, "camera_motion": False}
    train_records = [rgb_motion_world.episode(
        31000 + i, steps=context + horizon, size=size, **world)
        for i in range(train_count)]
    val_records = [rgb_motion_world.episode(
        61000 + i, steps=context + horizon, size=size, **world)
        for i in range(val_count)]
    train = bench.frames(train_records)
    val = bench.frames(val_records)
    fg_pool = foreground_episodes(train_records, context)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)

    results = []
    for arm in arms:
        for seed in seeds:
            params, loss_fn, rollout_fn, balanced = arm_specification(
                arm, jax.random.PRNGKey(int(seed)), width, context, horizon, ticks)
            moments = jax.tree.map(jnp.zeros_like, params)
            second = jax.tree.map(jnp.zeros_like, params)

            @jax.jit
            def update(p, m, v, batch, step):
                value, grad = jax.value_and_grad(loss_fn)(p, batch)
                norm = jnp.sqrt(sum(jnp.square(g).sum()
                                    for g in jax.tree.leaves(grad)))
                grad = jax.tree.map(lambda g: g / jnp.maximum(norm, 1.0), grad)
                p, m, v = adamw(p, grad, m, v, step)
                return p, m, v, value

            rng = np.random.default_rng(int(seed) + 7000)
            budget_reports = {}
            step = 0
            for target in budgets:
                while step < target:
                    step += 1
                    if balanced:
                        # Balanced sampling: at least one foreground episode.
                        idx = rng.integers(0, len(train), size=batch_size)
                        idx[0] = fg_pool[rng.integers(0, len(fg_pool))]
                    else:
                        idx = rng.integers(0, len(train), size=batch_size)
                    params, moments, second, _ = update(
                        params, moments, second, jnp.asarray(train[idx]), step)
                report = validation_report(
                    rollout_fn, params, val_records, val, context, horizon)
                report["status"] = classify(report)
                budget_reports[str(target)] = report
                np.savez_compressed(
                    folder / f"{arm}-seed{seed}-step{target}.npz",
                    **jax.tree.map(np.asarray, params))
            results.append({"arm": arm, "seed": int(seed),
                            "budgets": budget_reports})

    report = {
        "experiment": "training_budget_sweep_v1",
        "arms": list(arms), "seeds": list(map(int, seeds)),
        "budgets": list(budgets), "context": context, "horizon": horizon,
        "size": size, "width": width, "ticks": ticks,
        "train_count": train_count, "val_count": val_count,
        "world": world,
        "anti_collapse_levers": {
            "nca": "control (collapsing baseline)",
            "nca_fg": "foreground-sensitive loss",
            "nca_blend": "non-saturating convex-blend output head",
            "nca_grounded": "blend head + foreground loss + balanced sampling"},
        "results": results,
        "limits": "CPU-scale budgets and small worlds. Validation foreground and "
                  "localization come from palette/visibility truth read only "
                  "after each forecast. Status labels (undertrained/collapsed/"
                  "learning) are heuristic thresholds on collapse diagnostics and "
                  "foreground recall, not a significance test. Checkpoints at each "
                  "budget allow re-scoring. No result is promoted to the README.",
    }
    (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/training-budget-sweep")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 41])
    p.add_argument("--budgets", nargs="+", type=int, default=[300, 600, 1200])
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    a = p.parse_args()
    report = run(a.out, tuple(a.arms), tuple(a.seeds), tuple(a.budgets),
                 a.context, a.horizon, a.size, a.width, batch_size=a.batch_size)
    summary = {r["arm"] + "-" + str(r["seed"]):
               {b: v["status"] for b, v in r["budgets"].items()}
               for r in report["results"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
