"""Minimal schema checks for a proposed research contract, not result verification."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected research contract schema_version=1")
    if data.get("status") not in ("proposed", "frozen", "amended"):
        raise ValueError("Unknown contract status")
    task = data.get("task", {})
    if task.get("input") != "past RGB only":
        raise ValueError("Primary benchmark must use past RGB only")
    for field in ("context_frames", "horizon_frames", "localization_tolerance_px"):
        number = task.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or number <= 0:
            raise ValueError("Invalid positive task field: " + field)
    if task.get("statistical_unit") != "scene_seed_pair":
        raise ValueError("Paired histories must form one statistical unit")
    if set(x.get("id") for x in data.get("hypotheses", [])) != {"H1", "H2", "H3", "H4"}:
        raise ValueError("Expected explicit H1–H4 hypotheses")
    roots = list(data.get("split_seed_roots", {}).values())
    if len(roots) != 3 or len(set(roots)) != 3 or not all(type(n) is int for n in roots):
        raise ValueError("Split roots must be three distinct integers")
    bounds = data.get("proposed_decision_thresholds", {})
    for key in ("absolute_success_improvement", "max_inference_latency_ratio_for_efficiency_claim"):
        if isinstance(bounds.get(key), bool) or not isinstance(bounds.get(key), (int, float)) or bounds[key] <= 0:
            raise ValueError("Invalid decision bound " + key)
    if not 0 < bounds["absolute_success_improvement"] <= 1:
        raise ValueError("Success improvement must be a probability difference")
    if bounds["max_inference_latency_ratio_for_efficiency_claim"] < 1:
        raise ValueError("Cost ratio must be >= 1")
    if data.get("primary_comparator_rule") != "best_valid_nonprivileged_comparator_selected_on_validation_only":
        raise ValueError("Comparator selection must be frozen before final test")
    return data


def load(path):
    path = Path(path)
    raw = path.read_bytes()
    return validate(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="experiments/research_contract_v1.json")
    args = parser.parse_args()
    data, digest = load(args.contract)
    print(json.dumps({"status": data["status"], "sha256": digest,
                      "hypotheses": [x["id"] for x in data["hypotheses"]]}, indent=2))


if __name__ == "__main__":
    main()
