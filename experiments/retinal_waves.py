"""Stage 1: generate a bounded traveling-wave retinal curriculum."""
import argparse
import json
from pathlib import Path

import numpy as np

from src import physical_fields


def run(out="runs/retinal-waves", seed=11, size=24, frames=24):
    record = physical_fields.episode(seed, "wave", size, frames)
    observed = record["observed"]
    result = {
        "experiment": "retinal_waves_v1", "seed": seed, "size": size,
        "frames": frames, "kind": "wave",
        "peak_activity": float(np.max(np.abs(observed))),
        "mean_frame_energy": float(np.mean(observed ** 2)),
        "physical_parameters": record["parameters"],
        "training_inputs": "observed displacement only; velocity is withheld",
    }
    dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "observed.npy", observed.astype(np.float32))
    (dest / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/retinal-waves")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--size", type=int, default=24)
    p.add_argument("--frames", type=int, default=24)
    a = p.parse_args()
    print(json.dumps(run(a.out, a.seed, a.size, a.frames), indent=2))
