"""Compare an idealized two-state circuit cell to existing RGB forecasting controls.

One fixed local rule; a separately fitted *linear* 4-feature readout for each
ablation. No JAX training, no future leakage, no fitted material parameters.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from src import rgb_benchmark
from src import rgb_circuit_cell as cell


def experiment(seed=11, train_episodes=12, validation_episodes=4,
               test_episodes=12, context=8, horizon=10, size=32, ticks=2,
               checkpoint=None):
    if min(train_episodes, validation_episodes, test_episodes) < 1:
        raise ValueError("Every split needs at least one episode")
    if context < 2 or horizon < 1 or size < 16:
        raise ValueError("Invalid context/horizon/size")
    # Three disjoint groups, no checkpoint selection against the test scenes.
    start = int(seed) * 100000
    train = rgb_benchmark.frames(rgb_benchmark.scenes(
        start + 1000, train_episodes, context, horizon, size))
    validation = rgb_benchmark.frames(rgb_benchmark.scenes(
        start + 30000, validation_episodes, context, horizon, size))
    test = rgb_benchmark.frames(rgb_benchmark.scenes(
        start + 60000, test_episodes, context, horizon, size))
    leads = tuple(h for h in rgb_benchmark.LEADS if h <= horizon)
    if horizon not in leads:
        leads = tuple(sorted(set((*leads, horizon))))
    report = {"seed": seed,
              "train_seed_start": start + 1000,
              "validation_seed_start": start + 30000,
              "test_seed_start": start + 60000,
              "episodes": {"train": train_episodes, "validation": validation_episodes,
                           "test": test_episodes},
              "context": context, "horizon": horizon, "size": size,
              "ticks": ticks, "state_scalars_per_frame": int(size * size * 3 * 2),
              "per_pixel_analog_boundary": "optimistic: 2 site state values per RGB channel",
              "mode_metrics": {}, "heads": {}, "validation_metrics": {}}
    for mode in cell.MODES:
        head = cell.fit_head(train, context, mode, ticks)
        report["heads"][mode] = head.tolist()
        predicted = cell.rollout(test[:, :context], head, horizon, mode, ticks)
        report["mode_metrics"][mode] = rgb_benchmark.score(predicted, test, context, leads)
        val = cell.rollout(validation[:, :context], head, horizon, mode, ticks)
        report["validation_metrics"][mode] = rgb_benchmark.score(val, validation, context, leads)
    for name, forecast in rgb_benchmark.controls(test[:, :context], horizon).items():
        report["mode_metrics"][name] = rgb_benchmark.score(forecast, test, context, leads)

    if checkpoint:
        # Frozen external NCA; no fitting to these evaluation scenes.
        import jax.numpy as jnp
        from src import predictive_rgb_nca as nca
        with np.load(checkpoint, allow_pickle=False) as data:
            params = {key: jnp.asarray(data[key]) for key in data.files}
        forecast = np.asarray(nca.rollout(params, jnp.asarray(test[:, :context]),
                                          horizon, ticks))
        report["mode_metrics"]["frozen_rgb_nca"] = rgb_benchmark.score(
            forecast, test, context, leads)
        report["checkpoint"] = str(checkpoint)
    report["limitations"] = (
        "Idealized equation, not calibrated graphene hardware. Each RGB pixel/channel "
        "has independent v,m and receives a perfect drive; a full per-pixel digital "
        "readout is expensive. The same four-term linear head is fitted per arm. "
        "No energy, ADC/DAC, physical timing, real-webcam or causality claims. "
        "Disjoint synthetic episode seeds do not prove out-of-distribution transfer."
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--train-episodes", type=int, default=12)
    p.add_argument("--validation-episodes", type=int, default=4)
    p.add_argument("--test-episodes", type=int, default=12)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default="runs/circuit-rgb-comparison/metrics.json")
    a = p.parse_args()
    result = experiment(a.seed, a.train_episodes, a.validation_episodes,
                        a.test_episodes, a.context, a.horizon, a.size, a.ticks,
                        a.checkpoint)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mode_metrics"], indent=2))


if __name__ == "__main__":
    main()
