"""Combine saved experiment metrics without mixing incompatible protocols.

No training occurs. A manifest supplies exact JSON metric paths and identity
fields; grouping is strict and provenance is explicit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

FIELDS = ("name", "technique", "representation", "benchmark", "metric",
          "unit", "direction", "horizon_frames", "split", "seed_protocol",
          "checkpoint_selection", "source", "metric_path")
COMPARABLE = ("representation", "benchmark", "metric", "unit", "direction",
              "horizon_frames", "split", "seed_protocol", "checkpoint_selection")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nested(data, path):
    """Read dot-separated dictionary keys or numeric list indices."""
    result = data
    for token in path.split("."):
        if isinstance(result, dict):
            if token not in result:
                raise ValueError(f"Missing metric key {token!r} in {path!r}")
            result = result[token]
        elif isinstance(result, list) and token.isdecimal():
            index = int(token)
            if index >= len(result):
                raise ValueError(f"Index {index} absent in {path!r}")
            result = result[index]
        else:
            raise ValueError(f"Cannot follow {path!r} at {token!r}")
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"Metric {path!r} must be a numeric scalar")
    value = float(result)
    if not math.isfinite(value):
        raise ValueError(f"Metric {path!r} is not finite")
    return value


def collect(manifest, allow_missing=False):
    """Artifact paths are interpreted relative to current working directory."""
    content = json.loads(Path(manifest).read_text())
    items = content.get("entries")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a nonempty entries list")
    evidence, missing = [], []
    seen = set()
    for entry in items:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest entry must be an object")
        absent = [key for key in FIELDS if key not in entry]
        if absent:
            raise ValueError(f"Evidence entry missing required fields: {absent}")
        if entry["name"] in seen:
            raise ValueError(f"Duplicate evidence name: {entry['name']!r}")
        seen.add(entry["name"])
        if entry["direction"] not in ("lower", "higher"):
            raise ValueError("Metric direction must be lower or higher")
        if not isinstance(entry["horizon_frames"], int) or entry["horizon_frames"] < 1:
            raise ValueError("horizon_frames must be a positive integer")
        for key in FIELDS:
            if isinstance(entry[key], str) and not entry[key].strip():
                raise ValueError(f"Empty field {key!r}")
        source = Path(entry["source"])
        if not source.is_file():
            if not allow_missing:
                raise FileNotFoundError(source)
            missing.append({"name": entry["name"], "source": str(source),
                            "reason": "metrics artifact not present"})
            continue
        value = nested(json.loads(source.read_text()), entry["metric_path"])
        checkpoint = entry.get("checkpoint")
        checkpoint_digest = None
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_file():
                checkpoint_digest = file_hash(checkpoint_path)
            elif not allow_missing:
                raise FileNotFoundError(checkpoint_path)
        record = {key: entry[key] for key in FIELDS}
        record.update(value=value, source_sha256=file_hash(source),
                      checkpoint_sha256=checkpoint_digest,
                      revision=entry.get("revision"),
                      notes=entry.get("notes", ""),
                      provenance_status=(
                          "verified-metrics-and-checkpoint"
                          if checkpoint_digest else
                          "metrics-only-unverified-checkpoint"))
        evidence.append(record)
    grouped = {}
    for entry in evidence:
        key = tuple(entry[field] for field in COMPARABLE)
        grouped.setdefault(key, []).append(entry)
    return {
        "manifest": str(manifest), "evidence": evidence, "missing": missing,
        "groups": [{
            "comparison": dict(zip(COMPARABLE, key)),
            "entries": [row["name"] for row in rows],
            "comparable": len(rows) >= 2}
            for key, rows in grouped.items()],
        "interpretation": "Only measurements sharing representation, benchmark, "
                          "metric, unit, direction, horizon, split, seed "
                          "protocol and checkpoint-selection rule may be "
                          "compared. No global winner is inferred.",
    }


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown(report):
    """Readable audit with no cross-representation rankings."""
    lines = [
        "# Predictive Mirror - Cross-technique evidence", "",
        "A collection of measurements, **not a leaderboard**. Different",
        "representations, units, horizons and protocols are not ranked.", "",
        f"Records: **{len(report['evidence'])}**; "
        f"comparison groups: **{len(report['groups'])}**; "
        f"missing artifacts: **{len(report['missing'])}**.", ""]
    lookup = {entry["name"]: entry for entry in report["evidence"]}
    for index, group in enumerate(report["groups"], 1):
        key = group["comparison"]
        lines.extend([
            f"## Group {index}: {escape(key['representation'])} - "
            f"{escape(key['benchmark'])}", "",
            f"Metric: **{escape(key['metric'])}** "
            f"({escape(key['unit'])}, {escape(key['direction'])} is better); "
            f"+{key['horizon_frames']} frames, {escape(key['split'])}; "
            f"selection: {escape(key['checkpoint_selection'])}.", "",
            "| Technique | Evidence | Value | Provenance |",
            "|---|---|---:|---|"])
        for name in group["entries"]:
            entry = lookup[name]
            lines.append(
                f"| {escape(entry['technique'])} | {escape(name)} | "
                f"{entry['value']:.8g} | {escape(entry['provenance_status'])} |")
        lines.extend(["", "Comparable within declared protocol."
                      if group["comparable"] else
                      "No same-protocol comparator in this manifest.", ""])
    if report["missing"]:
        lines.extend(["## Missing artifacts", ""])
        for item in report["missing"]:
            lines.append(f"- {escape(item['name'])}: "
                         f"{escape(item['source'])} (not measured)")
        lines.append("")
    lines.extend(["## Audit notes", ""])
    for entry in report["evidence"]:
        lines.append(
            f"- **{escape(entry['name'])}**: source {escape(entry['source'])}, "
            f"SHA256 {entry['source_sha256']}; "
            + (f"checkpoint SHA256 {entry['checkpoint_sha256']}"
               if entry["checkpoint_sha256"] else "checkpoint not verified")
            + (f"; revision {escape(entry['revision'])}"
               if entry["revision"] else "; revision not recorded") + ".")
    lines.extend(["", escape(report["interpretation"]), ""])
    return "\n".join(lines)


def run(manifest, out="runs/evidence-report", allow_missing=False):
    report = collect(manifest, allow_missing)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"evidence.json").write_text(json.dumps(report, indent=2)+"\n")
    (folder/"report.md").write_text(markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="runs/evidence-report")
    parser.add_argument("--allow-missing", action="store_true")
    a = parser.parse_args()
    report = run(a.manifest, a.out, a.allow_missing)
    print(json.dumps({"records": len(report["evidence"]),
                      "groups": len(report["groups"]),
                      "missing": len(report["missing"]),
                      "report": str(Path(a.out)/"report.md")}, indent=2))


if __name__ == "__main__":
    main()
