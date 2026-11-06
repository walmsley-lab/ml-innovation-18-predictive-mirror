"""Stage 4: interact with perspective 3D views through the same retinal sheet."""
import argparse
import json
from pathlib import Path

from experiments.developmental_interaction_path import run as run_path


def run(out="runs/active-3d-learning", seed=11, frames=24, size=32, sheet=8):
    result = run_path(out, seed, frames, size, sheet, .04, "world3d")
    result["experiment"] = "active_3d_learning_v1"
    result["interaction"] = ["perspective", "camera actions", "moving objects"]
    result["privileged_state"] = ["depth", "pose", "identity"]
    Path(out, "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/active-3d-learning")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--sheet", type=int, default=8)
    a = p.parse_args()
    print(json.dumps(run(a.out, a.seed, a.frames, a.size, a.sheet), indent=2))
