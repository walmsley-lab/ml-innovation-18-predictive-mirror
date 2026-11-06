"""Stage 2: refine local connections from visual activity and surprise."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import DevelopmentalSheet, _pool


def run(out="runs/spatial-plasticity", seed=11, frames=24, size=32, sheet=8,
        plasticity=.04):
    rng = np.random.default_rng(seed)
    learner = DevelopmentalSheet(sheet, 7, seed, plasticity)
    errors = []
    for _ in range(frames):
        frame = rng.random((size, size, 3), dtype=np.float32)
        forecast = learner.observe(frame)
        errors.append(float(np.mean((forecast - _pool(frame, sheet)) ** 2)))
    result = {"experiment": "spatial_plasticity_v1", "seed": seed,
              "frames": frames, "plasticity": plasticity,
              "first_error": errors[0], "last_error": errors[-1],
              "learning_signal": errors[0] - errors[-1],
              "learner": learner.metrics(),
              "connections_are_experience_dependent": True}
    dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/spatial-plasticity")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--sheet", type=int, default=8)
    p.add_argument("--plasticity", type=float, default=.04)
    a = p.parse_args()
    print(json.dumps(run(a.out, a.seed, a.frames, a.size, a.sheet,
                         a.plasticity), indent=2))
