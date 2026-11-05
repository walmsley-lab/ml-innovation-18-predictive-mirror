"""Verify numerical field worlds; this is a simulator, not learned physics."""
import argparse
import json
from pathlib import Path
from src.physical_fields import verify, episode


def run(out="runs/physical-fields/verification.json", seed=11, size=24,
        steps=24, kind="both", boundary="reflect"):
    if kind not in ("both", "diffusion", "wave"):
        raise ValueError("Unknown field kind")
    check = verify(seed, size, steps)
    kinds = ("diffusion", "wave") if kind == "both" else (kind,)
    rows = {}
    for field in kinds:
        result = episode(seed, field, size, steps, boundary=boundary)
        rows[field] = {"shape": list(result["states"].shape),
                       "state_channels": int(result["states"].shape[1]),
                       "parameters": result["parameters"],
                       "initial_mass": float(result["observed"][0].sum())}
    report = {"protocol": "validated-field-worlds-v1",
              "verification": check, "worlds": rows,
              "limits": "Finite differences are a reference discretization, "
                        "not proof of continuum equivalence. Wave state includes "
                        "velocity; no learned forecaster evaluated."}
    dest = Path(out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/physical-fields/verification.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--kind", choices=("both", "diffusion", "wave"), default="both")
    parser.add_argument("--boundary", choices=("reflect", "zero"), default="reflect")
    a = parser.parse_args()
    print(json.dumps(run(a.out, a.seed, a.size, a.steps, a.kind, a.boundary),
                     indent=2))


if __name__ == "__main__":
    main()
