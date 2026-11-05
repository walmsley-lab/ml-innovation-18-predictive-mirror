"""Create and verify a portable, hashed local evidence bundle.

Does not execute training, certify originality, or assign scientific validity.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def revision():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL).strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment():
    versions = {}
    for package in ("jax", "jaxlib", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": versions}


def archive(metrics, checkpoints=(), out="runs/evidence-bundle",
            label="fresh-reproduction", command=None, selection=None):
    if label not in ("fresh-reproduction", "historical-unverified"):
        raise ValueError("Unknown evidence label")
    sources = [("metrics", Path(metrics))]
    sources.extend(("checkpoint", Path(x)) for x in checkpoints)
    if not all(p.is_file() for _, p in sources):
        raise FileNotFoundError("All declared evidence files must exist")
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Evidence bundle destination must be empty")
    (folder / "artifacts").mkdir(parents=True, exist_ok=True)
    records = []
    for index, (kind, source) in enumerate(sources):
        dest = folder / "artifacts" / f"{index:02d}-{source.name}"
        shutil.copy2(source, dest)
        records.append({"kind": kind, "name": source.name,
                        "path": dest.relative_to(folder).as_posix(),
                        "bytes": dest.stat().st_size, "sha256": digest(dest)})
    report = {
        "schema_version": 1, "status": label,
        "attestation": "Local files hashed and copied; scientific validity, "
                       "original provenance and independent reproduction NOT certified.",
        "source": revision(), "environment": environment(),
        "run_command": command, "selection_rule": selection,
        "artifacts": records
    }
    (folder / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify(folder):
    folder = Path(folder).resolve()
    report = json.loads((folder / "manifest.json").read_text())
    if report.get("schema_version") != 1 or not report.get("artifacts"):
        raise ValueError("Invalid evidence bundle manifest")
    seen = set()
    for item in report["artifacts"]:
        rel = Path(item["path"])
        if rel.is_absolute() or rel.parts[0] != "artifacts" or ".." in rel.parts:
            raise ValueError("Unsafe evidence artifact path")
        path = (folder / rel).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError("Missing or escaping evidence artifact")
        if str(rel) in seen:
            raise ValueError("Duplicate evidence artifact")
        seen.add(str(rel))
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Artifact mismatch: {item['path']}")
    return {"status": "hashes-match-local-artifacts",
            "file_count": len(seen),
            "independent_reproduction_verified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--out", default="runs/evidence-bundle")
    p.add_argument("--label", choices=("fresh-reproduction", "historical-unverified"),
                   default="fresh-reproduction")
    p.add_argument("--command", help="Literal run command for provenance; no execution")
    p.add_argument("--selection", help="Validation selection criterion")
    p.add_argument("--verify", help="Verify an existing bundle instead of creating one")
    a = p.parse_args()
    if a.verify:
        print(json.dumps(verify(a.verify), indent=2))
    else:
        if not a.metrics:
            p.error("--metrics is required when creating a bundle")
        result = archive(a.metrics, a.checkpoint, a.out, a.label,
                         a.command, a.selection)
        print(json.dumps({"bundle": a.out, "status": result["status"],
                          "artifacts": len(result["artifacts"])}, indent=2))


if __name__ == "__main__":
    main()
