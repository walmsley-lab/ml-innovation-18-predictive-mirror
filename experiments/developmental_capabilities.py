"""Validate the capability charter; no models are trained by this runner."""
import argparse
import json
from pathlib import Path

REQUIRED = {"id", "capability", "observations", "measures", "controls",
            "failure", "minimum_useful_effect", "species", "evidence", "falsifier"}


def validate(document):
    if document.get("schema") != "predictive-mirror-capabilities-v1":
        raise ValueError("Unknown capability schema")
    entries = document.get("capabilities", [])
    if not entries:
        raise ValueError("Empty capability list")
    seen = set()
    for item in entries:
        if not REQUIRED.issubset(item) or not all(item[key] for key in REQUIRED):
            raise ValueError("Incomplete capability")
        if item["id"] in seen:
            raise ValueError("Duplicate capability identifier")
        seen.add(item["id"])
        if any("future" in str(part).lower() or "oracle" in str(part).lower()
               for part in item["observations"]):
            raise ValueError("Future or privileged input cannot be a model observation")
    return {"capabilities": len(entries), "identifiers": sorted(seen),
            "confirmatory_results": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", default="experiments/developmental_capabilities.json")
    args = p.parse_args()
    print(json.dumps(validate(json.loads(Path(args.spec).read_text())), indent=2))


if __name__ == "__main__":
    main()
