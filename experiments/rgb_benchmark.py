"""Evaluate a frozen RGB NCA checkpoint on held-out synthetic scenes."""
import argparse
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark


def evaluate(checkpoint, output, seed=71000, episodes=12, context=8,
             horizon=10, size=32, ticks=2, optical_flow=False):
    x = benchmark.frames(benchmark.scenes(seed, episodes, context, horizon, size))
    with np.load(checkpoint, allow_pickle=False) as archive:
        params = {k: jnp.asarray(archive[k]) for k in archive.files}
    predicted = np.asarray(nca.rollout(params, jnp.asarray(x[:, :context]),
                                       horizon, ticks))
    leads = tuple(h for h in benchmark.LEADS if h <= horizon)
    outcomes = benchmark.compare(predicted, x, context, leads, optical_flow)
    report = {"test_seed_start": seed, "episodes": episodes,
              "context": context, "horizon": horizon,
              "checkpoint": str(checkpoint), "metrics": outcomes,
              "wins_at_horizon": {
                  name: bool(outcomes["nca"][str(horizon)]["mse"] <
                             outcomes[name][str(horizon)]["mse"])
                  for name in outcomes if name != "nca"
              } if horizon in leads else {},
              "caveat": "Test seeds must be excluded from training and checkpoint "
                        "selection. Predictions use observed context only; "
                        "future images used solely for scoring."}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/rgb-benchmark/metrics.json")
    p.add_argument("--seed", type=int, default=71000)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--optical-flow", action="store_true")
    a = p.parse_args()
    print(json.dumps(evaluate(a.checkpoint, a.out, a.seed, a.episodes, a.context,
                              a.horizon, a.size, a.ticks, a.optical_flow), indent=2))


if __name__ == "__main__":
    main()
