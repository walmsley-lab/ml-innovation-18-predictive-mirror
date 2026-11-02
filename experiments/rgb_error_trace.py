"""Held-out RGB NCA within-episode plastic trace, no weight updates."""
import argparse
import hashlib
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from src import rgb_error_trace as model, rgb_benchmark as benchmark


def run(checkpoint, out="runs/rgb-error-trace", seed=71000,
        episodes=12, context=8, horizon=10, size=32, ticks=2,
        decay=.82, strength=.12):
    if episodes < 1 or context < 2 or horizon < 1 or ticks < 1:
        raise ValueError("Invalid evaluation settings")
    path = Path(checkpoint)
    with np.load(path, allow_pickle=False) as archive:
        params = {key: jnp.asarray(archive[key]) for key in archive.files}
    examples = benchmark.frames(benchmark.scenes(
        seed, episodes, context, horizon, size))
    leads = tuple(sorted({horizon, *(h for h in benchmark.LEADS if h <= horizon)}))
    controls = {name: benchmark.score(values, examples, context, leads)
                for name, values in benchmark.controls(
                    examples[:, :context], horizon).items()}
    forecasts = {
        mode: np.asarray(model.rollout(
            params, jnp.asarray(examples[:, :context]),
            horizon, ticks, mode, decay, strength))
        for mode in model.MODES}
    intact = forecasts["frozen"]
    results = {
        mode: {
            "test": benchmark.score(values, examples, context, leads),
            "mean_absolute_difference_from_frozen": float(
                np.abs(values - intact).mean())
        } for mode, values in forecasts.items()}
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "example.npz",
                        context=examples[0, :context],
                        future_truth=examples[0, context:],
                        **{mode: frames[0] for mode, frames in forecasts.items()})
    report = {
        "experiment": "frozen_rgb_nca_local_prediction_error_trace",
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed_start": seed, "episodes": episodes, "context": context,
        "horizon": horizon, "ticks": ticks, "decay": decay,
        "trace_strength": strength, "controls": controls, "arms": results,
        "limitations": "This is a short-lived local error trace in state, NOT "
                       "synaptic weight plasticity or learned Hebbian connections. "
                       "The model weights never change. Test seed exclusion must "
                       "be verified relative to original training. Future truth "
                       "is used only for evaluation; no within-dream updates."}
    (folder / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/rgb-error-trace")
    p.add_argument("--seed", type=int, default=71000)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--decay", type=float, default=.82)
    p.add_argument("--strength", type=float, default=.12)
    a = p.parse_args()
    report = run(a.checkpoint, a.out, a.seed, a.episodes, a.context,
                 a.horizon, a.size, a.ticks, a.decay, a.strength)
    print(json.dumps({"arms": list(report["arms"]),
                      "metrics": str(Path(a.out)/"metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
