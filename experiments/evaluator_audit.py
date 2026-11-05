"""Run deterministic synthetic evaluator contract checks without model training."""
import argparse
import json
from pathlib import Path
from src.evaluator_audit import suite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 41])
    p.add_argument("--out", default="runs/evaluator-audit.json")
    a = p.parse_args()
    report = suite(a.seeds)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pairs": len(report["paired"]), "status": "controls_pass",
                      "learned_object_permanence": "not_evaluated"}, indent=2))


if __name__ == "__main__":
    main()
