"""Three-arm RGB NCA selection experiment with frozen routing rules."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import selectable_rgb_nca as nca
from train import adamw
from video_train import clips, windows
from experiments.rgb_nca import read_recording


def experiment(path="recordings/rgb-webcam.npz", out="runs/nca-selectors",
               steps=30, seed=41, batch_size=2, context=6, horizon=5, ticks=2):
    if steps < 1 or batch_size < 1 or context < 2 or horizon < 1 or ticks < 1:
        raise ValueError("Invalid training settings")
    frames, _ = read_recording(path)
    size = frames.shape[1]
    length = context + horizon - 1
    train_idx, val_idx = windows(frames, length)
    heldout = clips(frames, val_idx, length)
    refs = nca.reference_map(size, size)
    initial = nca.init(jax.random.PRNGKey(seed))
    weights = {mode: initial for mode in nca.MODES}
    zero = lambda p: jax.tree.map(jnp.zeros_like, p)
    first = {name: zero(initial) for name in weights}
    second = {name: zero(initial) for name in weights}

    def make_update(mode):
        references = refs if mode == "reference" else None
        @jax.jit
        def update(params, m, v, batch, step):
            loss = lambda p: nca.loss(p, batch, context, horizon, ticks, mode, references)
            error, grads = jax.value_and_grad(loss)(params)
            norm = jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
            grads = jax.tree.map(lambda g: g / jnp.maximum(norm, 1), grads)
            params, m, v = adamw(params, grads, m, v, step)
            return params, m, v, error
        return update

    update = {mode: make_update(mode) for mode in nca.MODES}
    rng = np.random.default_rng(seed + 77)
    for step in range(1, steps + 1):
        chosen = rng.choice(train_idx, size=batch_size, replace=True)
        batch = jnp.asarray(clips(frames, chosen, length))
        for mode in nca.MODES:
            weights[mode], first[mode], second[mode], error = update[mode](
                weights[mode], first[mode], second[mode], batch, step)
        if step == steps or step % max(1, steps // 5) == 0:
            print(f"step={step} final_arm_loss={float(error):.6f}", flush=True)

    outcomes = {}
    for mode in nca.MODES:
        ref = refs if mode == "reference" else None
        predicted = np.asarray(nca.rollout(
            weights[mode], jnp.asarray(heldout[:, :context]),
            horizon, ticks, mode, ref))
        target = heldout[:, context:context+horizon]
        last = heldout[:, context-1:context]
        changed = np.mean(np.abs(target-last), axis=-1) > 0.03
        squared = np.mean(np.square(target-predicted), axis=-1)
        outcomes[mode] = {
            "mse": float(squared.mean()),
            "changed_pixel_mse": float(squared[changed].mean()) if changed.any() else None,
            "changed_pixels": int(changed.sum()),
        }
    hold = float(np.square(
        heldout[:, context:context+horizon]-heldout[:, context-1:context]).mean())
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    for mode, params in weights.items():
        np.savez_compressed(folder / f"{mode}.npz", **jax.tree.map(np.asarray, params))
    report = {
        "experiment": "independent_rgb_nca_selector_ablation",
        "selector_modes": list(nca.MODES), "horizon": horizon, "ticks": ticks,
        "steps_per_arm": steps, "seed": seed, "validation_windows": len(val_idx),
        "persistence_mse": hold, "arms": outcomes,
        "limitations": "References use an externally fixed mirror-pixel mapping, NOT learned object correspondences. Changed neighborhood density, compute and communication distances are not matched. Same-recording chronological validation only.",
    }
    (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="recordings/rgb-webcam.npz")
    parser.add_argument("--out", default="runs/nca-selectors")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(experiment(args.input, args.out, args.steps, args.seed,
                                args.batch_size, args.context, args.horizon,
                                args.ticks), indent=2))


if __name__ == "__main__":
    main()
