"""Frozen-checkpoint probes for actual causal dependence on RGB hidden state."""
import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src import rgb_hidden_interventions as probe
from src import rgb_benchmark as benchmark


def run(checkpoint, out="runs/rgb-hidden-interventions", seed=71000,
        episodes=12, context=8, horizon=10, size=32, ticks=2,
        intervene_at=0):
    if episodes < 1 or context < 2 or horizon < 1 or ticks < 1:
        raise ValueError("Invalid evaluation settings")
    if not 0 <= intervene_at < horizon:
        raise ValueError("Intervention must occur between context and last horizon")
    path = Path(checkpoint)
    with np.load(path, allow_pickle=False) as archive:
        params = {key: jnp.asarray(archive[key]) for key in archive.files}
    frames = benchmark.frames(
        benchmark.scenes(seed, episodes, context, horizon, size))
    inputs = jnp.asarray(frames[:, :context])
    leads = tuple(sorted({horizon, *(h for h in benchmark.LEADS if h <= horizon)}))
    controls = {name: benchmark.score(values, frames, context, leads)
                for name, values in benchmark.controls(frames[:, :context], horizon).items()}
    forecasts = {
        mode: np.asarray(probe.forecast(
            params, inputs, horizon, ticks, mode, intervene_at))
        for mode in probe.MODES
    }
    original = forecasts["intact"]
    scores = {}
    for mode, values in forecasts.items():
        affected = np.abs(values-original)
        scores[mode] = {
            "test": benchmark.score(values, frames, context, leads),
            "delta_from_intact_mean_abs_by_lead": {
                str(lead): float(affected[:, lead-1].mean()) for lead in leads},
            "delta_from_intact_max_abs": float(affected.max()),
        }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    # Saved observed/predicted/truth arrays can be visualized without rerunning
    # trained models. Truth is never passed to the model.
    np.savez_compressed(
        target/"example.npz",
        observed=frames[0, :context],
        truth=frames[0, context:context+horizon],
        **{mode: forecast[0] for mode, forecast in forecasts.items()})
    report = {
        "experiment": "frozen_rgb_nca_hidden_state_interventions",
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "test_seed_start": seed, "episodes": episodes,
        "context": context, "horizon": horizon, "ticks": ticks,
        "intervention_before_prediction_index": intervene_at,
        "controls": controls, "modes": scores,
        "caveat": "Only RGB context reaches the model; hidden-state reset or "
                  "spatial permutation occurs after observations at a specified "
                  "rollout point. Significant forecast change establishes causal "
                  "dependence, NOT beneficial memory. Untrained zero-readout "
                  "checkpoints will show no differences. Held-out status requires "
                  "test seeds excluded from training and checkpoint selection.",
    }
    (target/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/rgb-hidden-interventions")
    p.add_argument("--seed", type=int, default=71000)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--intervene-at", type=int, default=0)
    a = p.parse_args()
    result = run(a.checkpoint, a.out, a.seed, a.episodes, a.context,
                 a.horizon, a.size, a.ticks, a.intervene_at)
    print(json.dumps({"modes": list(result["modes"]),
                      "results": str(Path(a.out)/"metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
