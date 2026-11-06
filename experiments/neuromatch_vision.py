"""Run four observational probes on synthetic RGB data or a frozen checkpoint."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_motion_world as world
from src import neuromatch_vision as nv


def run(parameters=None, seed=11, size=16, horizon=3, ticks=2):
    if size < 8 or horizon < 1:
        raise ValueError("Invalid size or horizon")
    frames = np.asarray(world.episode(seed + 70000, steps=9 + horizon,
                            size=size, balls=1, occlusion=True)["frames"])
    context, truth = frames[:9][None], frames[9:9+horizon][None]
    if parameters is None:
        params = nca.init(jax.random.PRNGKey(seed), channels=4)
        checkpoint = "random_untrained_control"
    else:
        with np.load(parameters, allow_pickle=False) as checkpoint_data:
            params = {k: jnp.asarray(checkpoint_data[k]) for k in checkpoint_data.files}
        checkpoint = str(parameters)
    observed = frames[7]
    sparse, costs = nv.sparse_observe(frames[6], observed)
    drift = np.clip(observed * 0.75, 0, 1)
    trajectory = nv.observed_trajectory(params, frames[:9], ticks=ticks)
    random_control = nca.init(jax.random.PRNGKey(seed+170), params["rgb_readout"].shape[0])
    untrained_trajectory = nv.observed_trajectory(random_control, frames[:9], ticks=ticks)
    return {
        "source": "synthetic_rgb_motion", "checkpoint": checkpoint, "seed": int(seed),
        "sparse": {**costs, "sparse_reconstruction_mse":
                   float(np.square(sparse - observed).mean()),
                   "dense_reconstruction_mse": 0.0},
        "normalization": {
            "raw_gain_shift_mae": float(np.abs(observed - drift).mean()),
            "normalized_gain_shift_mae": float(np.abs(
                nv.normalize_rgb(observed) - nv.normalize_rgb(drift)).mean()),
            "note": "Photometric sensitivity only; not evidence of improved prediction"},
        "dynamics": {
            "self_velocity_cka": nv.velocity_cka(trajectory, trajectory),
            "trained_vs_random_init_velocity_cka": nv.velocity_cka(
                trajectory, untrained_trajectory),
            "shuffled_velocity_cka": nv.velocity_cka(
                trajectory, trajectory[np.random.default_rng(seed).permutation(len(trajectory))]),
            "note": "Temporal velocity CKA proxy, not full DSA"},
        "lesion": nv.targeted_channel_lesions(params, context, truth,
                                              horizon=horizon, ticks=ticks),
        "limits": "A single synthetic example and optional frozen model; repeat across held-out seeds "
                  "and trained checkpoints before interpreting any differences."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--out", default="runs/neuromatch-vision/metrics.json")
    args = parser.parse_args()
    result = run(args.parameters, args.seed, args.size, args.horizon, args.ticks)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
