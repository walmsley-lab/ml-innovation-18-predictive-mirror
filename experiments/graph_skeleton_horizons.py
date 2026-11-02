"""Independent 10-step open-loop experiment for GIVEN 2D skeleton joints.

The original one-step skeleton NCA remains unchanged. Future 2D joint truth
is scoring-only, with visible future joints as the sole training targets.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.graph_skeleton_nca import examples
from src import graph_skeleton_nca as graph_nca
from train import adamw


def rollout(params, observed_context, visibility_context, horizon,
            graph, use_messages=True):
    """Absorb context, then dream without receiving future joint positions."""
    if horizon < 1 or observed_context.shape[1] < 2:
        raise ValueError("Need two context observations and positive horizon")
    batch, _, joints, _ = observed_context.shape
    state = graph_nca.initial_state(batch, joints, params["readout"].shape[0])

    def absorb(current, pair):
        return graph_nca.step(params, current, pair[0], pair[1],
                              graph, use_messages)
    state, _ = jax.lax.scan(absorb, state,
                           (jnp.swapaxes(observed_context, 0, 1),
                            jnp.swapaxes(visibility_context, 0, 1)))
    absent = jnp.zeros_like(observed_context[:, 0])
    no_visibility = jnp.zeros_like(visibility_context[:, 0])

    def dream(current, _):
        output = current[1]
        advanced, _ = graph_nca.step(
            params, current, absent, no_visibility, graph, use_messages)
        return advanced, output

    _, futures = jax.lax.scan(dream, state, None, length=horizon)
    return jnp.swapaxes(futures, 0, 1)


def loss(params, observed, visibility, context, horizon, graph, use_messages):
    predictions = rollout(params, observed[:, :context],
                          visibility[:, :context], horizon, graph, use_messages)
    target = observed[:, context:context+horizon]
    mask = visibility[:, context:context+horizon]
    error = jnp.sum(jnp.square(predictions-target), axis=-1, keepdims=True)
    return jnp.sum(mask*error)/jnp.maximum(mask.sum(), 1.0)


def last_seen(context, masks, horizon, use_velocity=False):
    """Observed-joint-only baseline, with optional per-joint velocity."""
    batch, length, nodes, _ = context.shape
    predictions = np.zeros((batch, horizon, nodes, 2), np.float32)
    for b in range(batch):
        last = np.zeros((nodes, 2), np.float32)
        previous = np.zeros_like(last)
        last_t = np.full(nodes, -1, int)
        previous_t = np.full(nodes, -1, int)
        for t in range(length):
            valid = masks[b, t, :, 0] > .5
            previous[valid] = last[valid]
            previous_t[valid] = last_t[valid]
            last[valid] = context[b, t, valid]
            last_t[valid] = t
        velocity = np.zeros_like(last)
        moving = (last_t >= 0) & (previous_t >= 0) & (last_t > previous_t)
        velocity[moving] = (last[moving]-previous[moving]) / (
            last_t[moving]-previous_t[moving])[:, None]
        for h in range(horizon):
            future_time = length+h
            predictions[b, h] = np.clip(
                last + (future_time-last_t)[:, None]*velocity if use_velocity else last,
                0, 1)
    return predictions


def score(predicted, truth, masks, previous_visibility, leads, pixel_scale=32):
    """Hidden positions enter only evaluation; count visible, hidden and reappearance."""
    result = {}
    past_mask = np.concatenate((previous_visibility[:, None], masks[:, :-1]), axis=1)
    for lead in leads:
        error = np.linalg.norm(predicted[:, lead-1]-truth[:, lead-1], axis=-1)*pixel_scale
        visible = masks[:, lead-1, :, 0] > .5
        reappearance = visible & (past_mask[:, lead-1, :, 0] < .5)
        hidden = ~visible
        result[str(lead)] = {
            "visible_joint_mae_px": float(error[visible].mean()) if visible.any() else None,
            "hidden_joint_mae_px": float(error[hidden].mean()) if hidden.any() else None,
            "reappearance_joint_mae_px": (
                float(error[reappearance].mean()) if reappearance.any() else None),
            "visible_targets": int(visible.sum()),
            "hidden_targets": int(hidden.sum()),
            "reappearance_events": int(reappearance.sum()),
        }
    return result


def train_ids(seed, step, batch_size):
    return 100_000+seed*100_000+step*batch_size+np.arange(batch_size)


def test_ids(seed, heldout):
    return 50_000_000+seed*1000+np.arange(heldout)


def check_disjoint(seeds, steps, batch_size, heldout):
    """Refuse settings where any training episode id is also a test id."""
    training = np.concatenate([train_ids(int(s), i, batch_size)
                               for s in seeds for i in range(1, steps+1)])
    testing = np.concatenate([test_ids(int(s), heldout) for s in seeds])
    if np.intersect1d(training, testing).size:
        raise ValueError("Training and held-out episode ids overlap; "
                         "use smaller seeds or fewer steps")


def run(output="runs/graph-skeleton-horizons", seeds=(11, 23, 37),
        steps=100, batch_size=4, heldout=12, context=8, horizon=10,
        occlusion_frames=3, channels=16):
    if not seeds or steps < 1 or batch_size < 1 or heldout < 1:
        raise ValueError("Invalid seed/training settings")
    if context < 2 or horizon < 1 or context+horizon-1 < 12:
        raise ValueError("Use context+horizon-1 >= 12 and positive context/horizon")
    check_disjoint(seeds, steps, batch_size, heldout)
    graph = graph_nca.adjacency()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    leads = tuple(sorted({h for h in (1, 3, 5, 10, horizon) if h <= horizon}))
    cases = []
    for seed in seeds:
        initial = graph_nca.init(jax.random.PRNGKey(seed), channels=channels)
        params = {"graph": initial, "no_messages": initial}
        moments = {k: jax.tree.map(jnp.zeros_like, initial) for k in params}
        variances = {k: jax.tree.map(jnp.zeros_like, initial) for k in params}

        def make_update(use_messages):
            @jax.jit
            def update(p, m, v, observed, visibility, step):
                err, grads = jax.value_and_grad(loss)(
                    p, observed, visibility, context, horizon, graph, use_messages)
                norm = jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grads)))
                grads = jax.tree.map(lambda g: g/jnp.maximum(1., norm), grads)
                p, m, v = adamw(p, grads, m, v, step)
                return p, m, v, err
            return update
        updates = {"graph": make_update(True), "no_messages": make_update(False)}
        for i in range(1, steps+1):
            episode_ids = train_ids(seed, i, batch_size)
            observed, mask, _ = examples(
                episode_ids, steps=context+horizon-1, occlusion_frames=occlusion_frames)
            errors = {}
            for name in params:
                params[name], moments[name], variances[name], errors[name] = updates[name](
                    params[name], moments[name], variances[name],
                    jnp.asarray(observed), jnp.asarray(mask), i)
            if i == steps or i % max(1, steps//5) == 0:
                print(f"seed={seed} step={i} "
                      + " ".join(f"{k}_loss={float(v):.6f}" for k, v in errors.items()),
                      flush=True)

        observed, mask, truth = examples(
            test_ids(seed, heldout), steps=context+horizon-1, occlusion_frames=occlusion_frames)
        future_truth = truth[:, context:context+horizon]
        future_masks = mask[:, context:context+horizon]
        prev_mask = mask[:, context-1]
        metrics = {}
        controls = {"last_seen": last_seen(observed[:, :context], mask[:, :context], horizon),
                    "constant_velocity": last_seen(
                        observed[:, :context], mask[:, :context], horizon, True)}
        for name, prediction in controls.items():
            metrics[name] = score(prediction, future_truth, future_masks, prev_mask, leads)
        for name, use_messages in (("graph", True), ("no_messages", False)):
            predicted = np.asarray(rollout(
                params[name], jnp.asarray(observed[:, :context]),
                jnp.asarray(mask[:, :context]), horizon, graph, use_messages))
            metrics[name] = score(predicted, future_truth, future_masks, prev_mask, leads)
            np.savez_compressed(output/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray, params[name]))
        cases.append({"seed": int(seed), "metrics": metrics})
    result = {"experiment": "oracle_2d_skeleton_10_step_open_loop",
              "seeds": list(map(int, seeds)), "context": context,
              "horizon": horizon, "steps_per_arm": steps,
              "occlusion_frames": occlusion_frames, "leads": list(leads),
              "runs": cases,
              "checkpoint_selection": "none: metrics use final-step weights, so "
                                      "held-out scenes never influence selection",
              "limitations": "Provided 2D joints and known topology (not RGB pose extraction). "
                             "Future observations are never inputs. Hidden-joint truth is scoring-only; "
                             "future visible joints alone train the predictor. Fixed synthetic actors "
                             "and simple velocity baselines; no claim of 3D or object discovery."}
    (output/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/graph-skeleton-horizons")
    p.add_argument("--seeds", nargs="+", type=int, default=(11, 23, 37))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--heldout", type=int, default=12)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--occlusion-frames", type=int, choices=(1, 3, 5), default=3)
    p.add_argument("--channels", type=int, default=16)
    a = p.parse_args()
    report = run(a.out, a.seeds, a.steps, a.batch_size, a.heldout,
                 a.context, a.horizon, a.occlusion_frames, a.channels)
    print(json.dumps({"seeds": report["seeds"],
                      "metrics_file": str(Path(a.out)/"metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
