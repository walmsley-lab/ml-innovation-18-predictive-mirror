"""Frozen RGB NCA harmonic microscope: checkpoint, step-zero, and shuffle."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark
from src import rgb_harmonics as harmonics


def run(checkpoint, out="runs/rgb-harmonic-microscope", seed=71000,
        episodes=4, context=8, horizon=10, size=32, ticks=2, modes=16):
    if episodes < 1 or context < 2 or horizon < 1 or ticks < 1 or modes < 1:
        raise ValueError("Invalid settings")
    path = Path(checkpoint)
    with np.load(path, allow_pickle=False) as f:
        params = {key: jnp.asarray(f[key]) for key in f.files}
    channels = params["rgb_readout"].shape[0]
    untrained = nca.init(jax.random.PRNGKey(seed + 1), channels)
    samples = benchmark.frames(
        benchmark.scenes(seed, episodes, context, horizon, size))
    observed = jnp.asarray(samples[:, :context])
    eigvals, eigmodes = harmonics.lattice_modes(size, size, modes)
    result = {}
    traces = {}
    for name, weights in (("checkpoint", params), ("random_initialization", untrained)):
        seen, dreamed, predicted = harmonics.trace(weights, observed, horizon, ticks)
        traces[name] = (np.asarray(seen), np.asarray(dreamed))
        result[name] = {
            "observed": harmonics.spectrum(seen, eigmodes),
            "dreamed": harmonics.spectrum(dreamed, eigmodes),
            "forecast_mse": float(np.square(
                np.asarray(predicted) - samples[:, context:context + horizon]).mean()),
        }
    result["checkpoint_shuffled"] = {
        "observed": harmonics.spectrum(
            harmonics.shuffle_sites(traces["checkpoint"][0], seed + 2), eigmodes),
        "dreamed": harmonics.spectrum(
            harmonics.shuffle_sites(traces["checkpoint"][1], seed + 2), eigmodes),
        "forecast_mse": None,  # Analysis-only shuffle, not a model intervention.
    }
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "example.npz",
        observed=samples[0, :context], future_truth=samples[0, context:],
        observed_hidden=traces["checkpoint"][0][0],
        dreamed_hidden=traces["checkpoint"][1][0],
        eigenvalues=eigvals, modes=eigmodes)
    report = {
        "experiment": "rgb_nca_harmonic_microscope",
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "episodes": episodes, "seed_start": seed, "context": context,
        "horizon": horizon, "ticks": ticks, "size": size,
        "requested_modes": modes, "actual_modes_including_ties": len(eigvals),
        "eigenvalues": eigvals.tolist(), "arms": result,
        "limits": "Exploratory spatial energy decomposition, not an observed "
                  "neural wave or learned representation. A fixed, orthogonal "
                  "basis can describe any field. Spatial shuffle affects "
                  "analysis only, not forecast scores. Checkpoint provenance "
                  "and held-out episode seeds must be verified externally.",
    }
    (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/rgb-harmonic-microscope")
    p.add_argument("--seed", type=int, default=71000)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--modes", type=int, default=16)
    a = p.parse_args()
    r = run(a.checkpoint, a.out, a.seed, a.episodes, a.context, a.horizon,
            a.size, a.ticks, a.modes)
    print(json.dumps({k: v["dreamed"]["selected_fraction"]
                      for k, v in r["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
