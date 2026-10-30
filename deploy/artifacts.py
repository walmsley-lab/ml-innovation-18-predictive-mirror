"""Export and restore synthetic Akash artifacts with SHA-256 verification."""

import argparse
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


def safe_relative(name):
    """A relative artifact name, without parent traversal or URL escapes."""
    path = Path(name)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts):
        raise ValueError(f"Invalid artifact path: {name}")
    if path.parts[0] not in ("pilot", "sweep"):
        raise ValueError("Only synthetic pilot/ and sweep/ artifacts may be exported")
    return path


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(base_url, name):
    """Read an exact artifact, using URL-encoded relative components."""
    rel = safe_relative(name)
    path = "/".join(urllib.parse.quote(part, safe="") for part in rel.parts)
    with urllib.request.urlopen(base_url.rstrip("/") + "/" + path, timeout=30) as response:
        return response.read()


def export(base_url, destination, paths):
    """Capture explicitly requested paths; verify checkpoint against remote sidecar."""
    archive = Path(destination)
    archive.mkdir(parents=True, exist_ok=True)
    records = []
    for raw in paths:
        rel = safe_relative(raw)
        content = fetch(base_url, str(rel))
        downloads = [(rel, content)]
        if rel.name == "checkpoint.npz":
            checksum_rel = safe_relative(str(rel) + ".sha256")
            reference_bytes = fetch(base_url, str(checksum_rel))
            reference = reference_bytes.decode().split()[0]
            if hashlib.sha256(content).hexdigest() != reference:
                raise ValueError(f"Live checkpoint changed during export: {rel}; retry")
            downloads.append((checksum_rel, reference_bytes))
        for artifact, payload in downloads:
            target = archive / artifact
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            temporary.write_bytes(payload)
            os.replace(temporary, target)
            records.append({"path": str(artifact), "sha256": hash_file(target),
                            "bytes": len(payload)})
    manifest = archive / "manifest.json"
    temporary = archive / "manifest.json.part"
    temporary.write_text(json.dumps({"version": 1, "files": records}, indent=2) + "\n")
    os.replace(temporary, manifest)
    return records


def restore(archive, destination):
    """Check the archive before copying and refuse conflicting destination files."""
    archive = Path(archive)
    items = json.loads((archive / "manifest.json").read_text())["files"]
    for item in items:
        rel = safe_relative(item["path"])
        source = archive / rel
        if not source.is_file() or hash_file(source) != item["sha256"]:
            raise ValueError(f"Backup checksum mismatch: {rel}")
    for item in items:
        rel = safe_relative(item["path"])
        source = archive / rel
        target = Path(destination) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hash_file(target) != item["sha256"]:
                raise FileExistsError(f"Refusing to overwrite changed file: {target}")
            continue
        shutil.copy2(source, target)
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--url", required=True, help="Akash public synthetic-artifact server URL")
    exp.add_argument("--out", required=True, help="Local backup directory")
    exp.add_argument("--path", action="append", required=True,
                     help="Relative artifact path, repeat to select several")
    res = sub.add_parser("restore")
    res.add_argument("--archive", required=True)
    res.add_argument("--dest", required=True, help="A fresh runs/ directory")
    args = parser.parse_args()
    if args.command == "export":
        if urllib.parse.urlparse(args.url).scheme not in ("https", "http"):
            parser.error("The provider URL must be HTTP(S)")
        result = export(args.url, args.out, args.path)
        print(json.dumps(result, indent=2))
    else:
        print(f"Restored {restore(args.archive, args.dest)} checked artifact(s)")


if __name__ == "__main__":
    main()
