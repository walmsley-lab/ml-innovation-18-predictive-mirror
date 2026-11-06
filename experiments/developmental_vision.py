"""Stage 3: online retinal prediction after wave and plasticity exposure."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.developmental_interaction_path import DevelopmentalSheet, _pool, _retinal_frames


def run(out="runs/developmental-vision", seed=11, frames=24, size=32, sheet=8):
    learner = DevelopmentalSheet(sheet, 7, seed, .04)
    wave, _ = _retinal_frames("wave", seed, frames // 2 + 1, size)
    visual, meta = _retinal_frames("world3d", seed + 1, frames - len(wave) + 1, size)
    errors = []
    sequence = np.concatenate((wave, visual[1:]))
    for index, frame in enumerate(sequence):
        offset = index - len(wave)
        action = meta["actions"][offset] if 0 <= offset < len(meta["actions"]) else (0., 0.)
        forecast = learner.observe(frame, action)
        errors.append(float(np.mean((forecast - _pool(frame, sheet)) ** 2)))
    result = {"experiment": "developmental_vision_v1", "seed": seed,
              "frames": len(errors), "stage_boundary": len(wave),
              "first_error": errors[0], "last_error": errors[-1],
              "learning_signal": errors[0] - errors[-1],
              "learner": learner.metrics(), "continuity": "single learner, no reset",
              "sensorimotor_actions_used": True}
    dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/developmental-vision")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--sheet", type=int, default=8)
    a = p.parse_args()
    print(json.dumps(run(a.out, a.seed, a.frames, a.size, a.sheet), indent=2))
