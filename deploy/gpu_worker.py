"""Private Akash campaign worker and artifact bridge; never opens an HTTP port.

The container stays alive for a bounded retrieval window after training.
The client reads status and checksummed file chunks through akt console shell.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEFAULT_OUTPUT = "/app/runs/campaign"
CHUNK_MAX = 96 * 1024


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_at(root, relative):
    """Resolve only plain files inside a configured, private output tree."""
    base = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts):
        raise ValueError("Unsafe relative artifact path")
    target = base.joinpath(path)
    if target.is_symlink() or not target.resolve().is_relative_to(base):
        raise ValueError("Artifact symlink or path escape")
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def inventory(root):
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    files = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in private output tree")
        if not path.is_file() or path.name.endswith((".part", ".tmp")):
            continue
        files.append({"path": path.relative_to(base).as_posix(),
                      "bytes": path.stat().st_size, "sha256": hash_file(path)})
    return {"version": 1, "files": files}


def read_chunk(root, relative, offset, length):
    if offset < 0 or not 1 <= length <= CHUNK_MAX:
        raise ValueError("Invalid chunk offset or length")
    file = file_at(root, relative)
    if offset > file.stat().st_size:
        raise ValueError("Chunk offset after EOF")
    with file.open("rb") as handle:
        handle.seek(offset)
        return base64.b64encode(handle.read(length)).decode("ascii")


def gpu_available():
    import jax
    devices = jax.devices()
    return any(d.platform == "gpu" for d in devices)


def run_campaign(config, output, max_runs, expected_sha, hold_seconds):
    """Run the repo's existing campaign.py and remain available to akt exec."""
    config = Path(config).resolve()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = out.parent / "gpu_status.json"
    state = {"status": "starting", "config_sha256": expected_sha,
             "source_revision": os.environ.get("SOURCE_REVISION", "unknown")}
    try:
        if not config.is_file() or hash_file(config) != expected_sha:
            raise ValueError("Container config differs from approved local manifest")
        if max_runs < 1 or not 60 <= hold_seconds <= 86400:
            raise ValueError("Invalid bounds")
        if not gpu_available():
            raise RuntimeError("JAX GPU device unavailable; refusing CPU fallback")
        state["status"] = "running"
        atomic_json(status, state)
        command = [sys.executable, "-m", "campaign", "--config", str(config),
                   "--output", str(out), "--execute", "--max-runs", str(max_runs),
                   "--require-gpu"]
        subprocess.run(command, check=True, cwd="/app")
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        atomic_json(status, state)
        print(json.dumps(state), flush=True)
    # Keep the lease reachable for private download. The client closes it
    # after export; Console runtime settings independently bound lease life.
    deadline = time.monotonic() + hold_seconds
    while time.monotonic() < deadline:
        time.sleep(max(0, min(30, deadline - time.monotonic())))
    return state["status"] == "completed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output", default=DEFAULT_OUTPUT)
    train.add_argument("--max-runs", type=int, required=True)
    train.add_argument("--sha256", required=True)
    train.add_argument("--hold-seconds", type=int, default=3600)
    for name in ("status", "inventory", "chunk"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", default=DEFAULT_OUTPUT)
        if name == "chunk":
            sub.add_argument("--path", required=True)
            sub.add_argument("--offset", type=int, required=True)
            sub.add_argument("--length", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        success = run_campaign(args.config, args.output, args.max_runs,
                               args.sha256, args.hold_seconds)
        return 0 if success else 1
    if args.command == "status":
        target = Path(args.output).parent / "gpu_status.json"
        print(target.read_text() if target.is_file()
              else json.dumps({"status": "starting"}))
    elif args.command == "inventory":
        print(json.dumps(inventory(args.output)))
    else:
        print(json.dumps({"data": read_chunk(args.output, args.path,
                                             args.offset, args.length)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
