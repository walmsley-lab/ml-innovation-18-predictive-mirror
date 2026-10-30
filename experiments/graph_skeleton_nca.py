"""Minimal 2D skeleton graph-NCA: learn, ablate messages, measure occlusion.

The supplied skeleton topology and 2D joints are *oracle representations*.
No RGB-to-pose estimator or 3D reconstruction is part of this experiment.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import graph_skeleton_nca as nca
from train import adamw


def episode(seed, steps=14, occlusion_frames=3):
    """Return visible observations, visibility, privileged truth for scoring.

    Only 2D coordinates and visibility may reach a model. Hidden truth is not
    passed to a model or used as a training label.
    """
    if steps < 12 or not 1 <= occlusion_frames <= 5:
        raise ValueError("steps >= 12 and occlusion_frames in 1..5")
    rng = np.random.default_rng(int(seed))
    root = np.array([rng.uniform(0.43, 0.57), rng.uniform(0.46, 0.53)], np.float32)
    phase = float(rng.uniform(0, 2 * np.pi))
    stride = float(rng.uniform(0.24, 0.56))
    amplitude = float(rng.uniform(0.014, 0.045))
    scale = float(rng.uniform(0.88, 1.10))
    poses = []
    for t in range(steps + 1):
        root_now = root + [0.045 * np.sin(0.17 * t + phase), 0.008 * np.sin(0.35 * t + phase)]
        pelvis = root_now
        neck = pelvis + [0.0, -0.17 * scale]
        left_shoulder = neck + [-0.07 * scale, 0.025 * scale]
        right_shoulder = neck + [0.07 * scale, 0.025 * scale]
        swing = amplitude * np.sin(stride * t + phase)
        left_elbow = left_shoulder + [-0.052 * scale - swing, 0.100 * scale]
        left_wrist = left_elbow + [-0.038 * scale - swing, 0.09 * scale]
        right_elbow = right_shoulder + [0.052 * scale - swing, 0.100 * scale]
        right_wrist = right_elbow + [0.038 * scale - swing, 0.09 * scale]
        left_knee = pelvis + [-0.052 * scale + swing, 0.14 * scale]
        left_ankle = left_knee + [-0.023 * scale + swing, 0.14 * scale]
        right_knee = pelvis + [0.052 * scale - swing, 0.14 * scale]
        right_ankle = right_knee + [0.023 * scale - swing, 0.14 * scale]
        poses.append(np.asarray((pelvis, neck, left_shoulder, left_elbow, left_wrist,
                                 right_shoulder, right_elbow, right_wrist,
                                 left_knee, left_ankle, right_knee, right_ankle),
                                np.float32))
    truth = np.stack(poses)
    visibility = np.ones((steps + 1, len(nca.JOINTS), 1), np.float32)
    visibility[2:2 + occlusion_frames, 4] = 0.0       # left wrist
    visibility[steps // 2:steps // 2 + occlusion_frames, 11] = 0.0  # right ankle
    observed = truth * visibility
    return observed, visibility, truth


def examples(seeds, steps=14, occlusion_frames=3):
    episodes = [episode(s, steps, occlusion_frames) for s in seeds]
    return tuple(np.stack([run[index] for run in episodes]) for index in range(3))


def persistence(observed, visibility):
    """Causal last-seen baseline: never reads hidden ground-truth coordinates."""
    last = np.zeros_like(observed[:, 0])
    results = []
    for t in range(observed.shape[1] - 1):
        mask = visibility[:, t]
        last = mask * observed[:, t] + (1.0 - mask) * last
        results.append(last.copy())
    return np.stack(results, axis=1)


def measure(predicted, truth, visibility, pixel_scale=32):
    """Scores use simulator truth ONLY after model inference, in evaluator."""
    error = np.linalg.norm(predicted - truth[:, 1:], axis=-1) * pixel_scale
    visible = visibility[:, 1:, :, 0] > 0.5
    hidden = ~visible
    reappearance = visible & (visibility[:, :-1, :, 0] < 0.5)
    return {"visible_joint_error_px": float(error[visible].mean()) if visible.any() else None,
            "hidden_joint_error_px": float(error[hidden].mean()) if hidden.any() else None,
            "reappearance_joint_error_px": float(error[reappearance].mean()) if reappearance.any() else None,
            "reappearance_events": int(reappearance.sum()),
            "visible_joint_targets": int(visible.sum())}


def experiment(output="runs/graph-skeleton-nca", seed=41, steps=100,
               batch_size=8, heldout=24, occlusion_frames=3):
    if steps < 1 or batch_size < 1 or heldout < 1:
        raise ValueError("steps, batch_size and heldout must be positive")
    graph = nca.adjacency()
    initial = nca.init(jax.random.PRNGKey(seed))
    params = {"graph": initial, "no_messages": initial}
    zeros = lambda p: jax.tree.map(jnp.zeros_like, p)
    first = {name: zeros(initial) for name in params}
    second = {name: zeros(initial) for name in params}

    def make_update(use_messages):
        @jax.jit
        def update(p, m, v, frames, masks, i):
            error, gradients = jax.value_and_grad(nca.loss)(p, frames, masks, graph, use_messages)
            p, m, v = adamw(p, gradients, m, v, i)
            return p, m, v, error
        return update

    updates = {"graph": make_update(True), "no_messages": make_update(False)}
    for i in range(1, steps + 1):
        seeds = seed * 1_000_000 + i * batch_size + np.arange(batch_size)
        frames, masks, _ = examples(seeds, occlusion_frames=occlusion_frames)
        x, v = jnp.asarray(frames), jnp.asarray(masks)
        for name in params:
            params[name], first[name], second[name], err = updates[name](
                params[name], first[name], second[name], x, v, i)
        if i == steps or i % max(steps // 5, 1) == 0:
            print(f"step={i} graph_loss={float(err):.6f}", flush=True)

    # Complete new actors (shapes, phases, and motion frequencies), no shared seeds.
    validation_seeds = 50_000_000 + seed * 1000 + np.arange(heldout)
    frames, masks, truth = examples(validation_seeds, occlusion_frames=occlusion_frames)
    result = {"experiment": "oracle_2d_skeleton_graph_nca", "seed": seed,
              "steps_per_arm": steps, "batch_size": batch_size,
              "heldout_actor_seeds": heldout, "joint_count": len(nca.JOINTS),
              "node_features": "2d visible coordinates, visibility bit, recurrent hidden state",
              "graph": "fixed undirected kinematic edges (given, not inferred)",
              "occlusion_frames": occlusion_frames,
              "representation": "oracle_2d_pose_no_3d_no_pixels",
              "arms": {}}
    result["arms"]["last_seen"] = measure(persistence(frames, masks), truth, masks)
    for name, has_messages in (("graph", True), ("no_messages", False)):
        forecast = np.asarray(nca.predict(params[name], jnp.asarray(frames),
                                          jnp.asarray(masks), graph, has_messages))
        result["arms"][name] = measure(forecast, truth, masks)
    result["caveats"] = ("Exploratory one-seed pilot; separately learnable 2D joint coordinates "
                         "are provided by the simulator, not recovered from camera video. "
                         "Only visible future joints train; hidden truth is evaluation-only. "
                         "No claim of emergent skeleton discovery or 3D reasoning.")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for name in params:
        np.savez_compressed(out / f"{name}.npz", **jax.tree.map(np.asarray, params[name]))
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/graph-skeleton-nca")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--heldout", type=int, default=24)
    p.add_argument("--occlusion-frames", type=int, choices=(1, 3, 5), default=3)
    args = p.parse_args()
    print(json.dumps(experiment(args.out, args.seed, args.steps, args.batch_size,
                                args.heldout, args.occlusion_frames), indent=2))


if __name__ == "__main__":
    main()
