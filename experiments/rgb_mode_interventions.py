"""Frozen-checkpoint causal suppression of lattice modes vs energy-matched null."""
import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src import rgb_benchmark as benchmark
from src import rgb_harmonics as harmonics
from src import rgb_mode_interventions as modes
from src import rgb_hidden_interventions as hidden
from src import forecast_diagnostics as diagnostics


def run(checkpoint, out="runs/rgb-mode-interventions", seed=71000,
        episodes=12, context=8, horizon=10, size=32, ticks=2,
        mode_count=16, intervene_at=0, strength=1.0):
    if (episodes < 1 or context < 2 or horizon < 1 or ticks < 1
            or mode_count < 1 or not 0 <= intervene_at < horizon
            or not 0 <= strength <= 1):
        raise ValueError("Invalid evaluation settings")
    path = Path(checkpoint)
    with np.load(path, allow_pickle=False) as f:
        params = {name: jnp.asarray(f[name]) for name in f.files}
    values = benchmark.frames(
        benchmark.scenes(seed, episodes, context, horizon, size))
    context_frames = jnp.asarray(values[:, :context])
    eigenvalues, basis = harmonics.lattice_modes(size, size, mode_count)
    null = modes.random_modes(size, size, len(eigenvalues), seed + 1337)
    leads = tuple(sorted({horizon, *(k for k in benchmark.LEADS if k <= horizon)}))
    predictions = {name: np.asarray(modes.forecast(
        params, context_frames, jnp.asarray(basis, jnp.float32),
        jnp.asarray(null), horizon, ticks, name, intervene_at, strength))
        for name in modes.MODES}
    intact = predictions["intact"]
    report_modes = {}
    for name, predicted in predictions.items():
        scoring = benchmark.score(predicted, values, context, leads)
        base_score = benchmark.score(intact, values, context, leads)
        report_modes[name] = {
            "test": scoring,
            "delta_mse_from_intact": {
                str(lead): float(scoring[str(lead)]["mse"]
                                 - base_score[str(lead)]["mse"])
                for lead in leads},
            "forecast_mean_absolute_change": float(
                np.abs(predicted - intact).mean()),
        }
    controls = {name: benchmark.score(pred, values, context, leads)
                for name, pred in benchmark.controls(
                    values[:, :context], horizon).items()}
    # Positive control: wipe the entire recurrent memory at the same lead. This
    # is the strongest possible hidden-state intervention, so if even it fails to
    # move the forecast at a lead, the mode-suppression null there is a dead
    # measurement (a collapsed, input-independent forecast), not a real null.
    wiped = np.asarray(hidden.forecast(
        params, context_frames, horizon, ticks, "reset", intervene_at))
    measurement = diagnostics.collapse_report(
        intact, leads, positive_control={"wipe_hidden_memory": wiped})
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "example.npz",
        observed=values[0, :context], future_truth=values[0, context:],
        **{name: pred[0] for name, pred in predictions.items()})
    report = {
        "experiment": "rgb_nca_causal_harmonic_mode_suppression",
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed_start": seed, "episodes": episodes, "context": context,
        "horizon": horizon, "size": size, "ticks": ticks,
        "intervene_before_prediction_index": intervene_at,
        "strength": strength, "requested_low_mode_count": mode_count,
        "selected_mode_count_including_ties": len(eigenvalues),
        "selected_eigenvalues": eigenvalues.tolist(),
        "interventions": report_modes, "baselines": controls,
        "measurement": measurement,
        "limits": "Paired intervention on one frozen model. Random-subspace "
                  "suppression matches the low-mode removal's per-sample, "
                  "per-channel L2 magnitude at intervention time. A forecast "
                  "difference alone is not evidence of improvement, semantic "
                  "representations, or brain mechanisms. Check that held-out "
                  "test seeds were excluded from checkpoint selection. The "
                  "'measurement' block reports whether the forecast has "
                  "collapsed to an input-independent clip-rail fixed point; at "
                  "any lead flagged 'dead' a ~0 suppression delta is "
                  "uninformative rather than a causal null.",
    }
    (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/rgb-mode-interventions")
    p.add_argument("--seed", type=int, default=71000)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--mode-count", type=int, default=16)
    p.add_argument("--intervene-at", type=int, default=0)
    p.add_argument("--strength", type=float, default=1.0)
    a = p.parse_args()
    report = run(a.checkpoint, a.out, a.seed, a.episodes, a.context,
                 a.horizon, a.size, a.ticks, a.mode_count,
                 a.intervene_at, a.strength)
    print(json.dumps({
        "delta_mse_from_intact": {name: arm["delta_mse_from_intact"] for name, arm
                                  in report["interventions"].items()},
        "measurement_verdict": report["measurement"].get("measurement_verdict"),
    }, indent=2))


if __name__ == "__main__":
    main()
