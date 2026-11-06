"""One-GPU Akash Console campaign lifecycle: plan, run, recover, close.

No deployment occurs without an explicit paid-run approval flag.
All remote outputs are retrieved over authenticated akt console shell,
not a publicly exposed HTTP endpoint.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import sys
import time

from deploy.akash_preflight import IMAGE, replace_once
from deploy.gpu_worker import CHUNK_MAX, hash_file

BASE = Path("runs/akash-gpu")
CONFIG_PATTERN = re.compile(r"experiments/[a-zA-Z0-9_-]+\.json\Z")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def cli(args, *, context=None):
    prefix = ["akt"] + (["--context", context] if context else [])
    result = subprocess.run(prefix + list(args), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True, timeout=120)
    return result.stdout


def load_manifest(path):
    relative = Path(path).as_posix()
    if not CONFIG_PATTERN.fullmatch(relative):
        raise ValueError("Manifest must be an embedded experiments/*.json path")
    manifest = Path(relative)
    payload = json.loads(manifest.read_text())
    if payload.get("kind") == "rgb_evidence_v1":
        from experiments.akash_rgb_evidence import validate
        return validate(payload), hash_file(manifest)
    required = {"seeds", "architectures", "curricula", "steps", "batch_size",
                "episodes", "checkpoint_every", "loss_mode", "primary_metric",
                "replay_fraction"}
    if set(payload) != required:
        raise ValueError("Manifest does not match existing campaign contract")
    for field in ("seeds", "architectures", "curricula"):
        if not isinstance(payload[field], list) or not payload[field]:
            raise ValueError("Campaign matrix dimension is empty")
    count = len(payload["seeds"]) * len(payload["architectures"]) * len(payload["curricula"])
    if count > 1000:
        raise ValueError("Refusing giant campaign")
    return count, hash_file(manifest)


def render_sdl(template, image, config, config_sha, count, hold_seconds):
    if not IMAGE.fullmatch(image):
        raise ValueError("Use verified immutable GHCR image digest")
    if not CONFIG_PATTERN.fullmatch(config):
        raise ValueError("Unsafe embedded config path")
    if not (1 <= count <= 1000 and 60 <= hold_seconds <= 86400):
        raise ValueError("Invalid remote campaign bounds")
    if not re.search(r"^\s+denom:\s*uact(?:\s*#.*)?$", template, re.MULTILINE):
        raise ValueError("Expected uact billing denomination")
    price = re.search(r"^\s+amount:\s*(\d+)", template, re.MULTILINE)
    if not price or not 1 <= int(price.group(1)) <= 2500:
        raise ValueError("Provider max bid exceeds reviewed 2500 uact/block")
    if not re.search(r"^\s+gpu:\s*\n\s+units:\s*1\s*$", template, re.MULTILINE):
        raise ValueError("Exactly one GPU required")
    if re.search(r"^\s+-\s+ARTIFACT_HTTP=1", template, re.MULTILINE):
        raise ValueError("Public artifact serving is forbidden for campaigns")
    output = replace_once(template, "image", r"^(\s+image:\s*)\S+(?:\s+#.*)?$",
                          lambda m: m.group(1) + image)
    for key, value in (("RUN_MODE", "campaign"), ("CAMPAIGN_APPROVED", "1"),
                       ("CAMPAIGN_MAX_RUNS", str(count)), ("REQUIRE_GPU", "1")):
        output = replace_once(output, key, rf"^(\s+-\s+{key}=).*$",
                              lambda m, v=value: m.group(1) + v)
    anchor = r"^(\s+-\s+CAMPAIGN_MAX_RUNS=.*)$"
    additions = (f"      - CAMPAIGN_CONFIG=/app/{config}\n"
                 f"      - CAMPAIGN_SHA256={config_sha}\n"
                 f"      - CAMPAIGN_HOLD_SECONDS={hold_seconds}\n"
                 "      - ARTIFACT_HTTP=0")
    output = replace_once(output, "campaign environment", anchor,
                          lambda m: m.group(1) + "\n" + additions)
    return output


def prepare(args):
    count, checksum = load_manifest(args.config)
    if count > args.max_runs:
        raise ValueError(f"Campaign has {count} runs; raise --max-runs explicitly")
    if not 1 <= args.hours <= 24:
        raise ValueError("Console lease TTL must be 1..24 hours")
    template = Path("deploy/akash-gpu.yaml").read_text()
    # The remote worker must remain reachable briefly after training for export.
    rendered = render_sdl(template, args.image, args.config, checksum, count,
                          min(args.hours * 3600, 3600))
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    sdl = root / "deployment.yaml"
    if sdl.exists() and sdl.read_text() != rendered:
        raise FileExistsError("Different reviewed SDL already exists: " + str(sdl))
    sdl.write_text(rendered)
    return sdl, {"config": args.config, "config_sha256": checksum, "image": args.image,
                 "run_count": count, "max_hours": args.hours, "out": str(root),
                 "bid_max_uact_per_block": 2500, "state": "planned"}


def console_check(context):
    # Fail closed if context is not the managed Console rail.
    details = cli(["context", "show", "-o", "json"], context=context)
    if "console-api" not in details:
        raise ValueError("Only akt console-api contexts are supported")
    cli(["console", "whoami", "-o", "json"], context=context)


def dseq_from_event(event):
    """Find dseq only in known structured workflow outputs."""
    outputs = event.get("outputs")
    if not isinstance(outputs, dict):
        return None
    value = outputs.get("dseq")
    if isinstance(value, (int, str)) and str(value).isdigit() and int(value) > 0:
        return str(value)
    return None


# akt may report the DSEQ only on stderr if a leased deployment times out
# during readiness. Capturing that ID is essential for the finally cleanup.
_FAILED_DEPLOY_DSEQ = re.compile(
    r"deployment DSEQ\s+([1-9][0-9]*)\s+with provider\b"
)


def dseq_from_failure_output(message):
    match = _FAILED_DEPLOY_DSEQ.search(message)
    return match.group(1) if match else None


def deploy(sdl, state, state_file, context):
    cmd = ["akt"] + (["--context", context] if context else []) + [
        "deploy", str(sdl), "--bid-select", "cheapest", "--ready-timeout", "10m",
        "--yes", "-o", "jsonl"]
    # A file rather than stderr=PIPE avoids deadlock while iterating stdout
    # from a CLI that may emit many progress/warning lines on stderr.
    with tempfile.TemporaryFile(mode="w+t") as error_log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=error_log,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        failed_event = False
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                found = dseq_from_event(event)
                if found:
                    if state.get("dseq") and state["dseq"] != found:
                        raise RuntimeError("Conflicting Akash deployment IDs")
                    state["dseq"] = found
                    atomic_json(state_file, state)
                if event.get("result") == "failed":
                    failed_event = True
            code = proc.wait()
            error_log.seek(0)
            errors = error_log.read()
            # The CLI can create a paid lease before it reports a readiness
            # failure. Store the DSEQ from its explicit cleanup warning.
            found = dseq_from_failure_output(errors)
            if found:
                if state.get("dseq") and state["dseq"] != found:
                    raise RuntimeError("Conflicting Akash deployment IDs")
                state["dseq"] = found
                atomic_json(state_file, state)
            if errors:
                print(errors, file=sys.stderr, end="" if errors.endswith("\n") else "\n")
            if code != 0 or failed_event:
                raise RuntimeError(
                    "Akash deployment workflow failed; captured DSEQ will be closed"
                    if state.get("dseq") else
                    "Akash deployment failed with unknown DSEQ; check active leases"
                )
            if not state.get("dseq"):
                raise RuntimeError("Missing dseq; inspect active deployments manually")
            return state["dseq"]
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)


def remote(dseq, command, context):
    raw = cli(["console", "shell", str(dseq), "trainer", "-o", "json", "--",
               "python", "-m", "deploy.gpu_worker"] + list(command),
              context=context)
    envelope = json.loads(raw)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("stdout"), str):
        raise RuntimeError("Unexpected akt console shell JSON response")
    return json.loads(envelope["stdout"].strip())


def safe_rel(name):
    path = Path(name)
    if not name or not path.parts or path.is_absolute() or any(p in ("..", ".") for p in path.parts):
        raise ValueError("Unsafe artifact name")
    return path


def download(state, context, max_bytes):
    dseq = state["dseq"]
    listing = remote(dseq, ["inventory"], context)
    files = listing["files"]
    if not isinstance(files, list) or sum(item["bytes"] for item in files) > max_bytes:
        raise ValueError("Artifact size exceeds approved download budget")
    dest = Path(state["out"]) / "artifacts"
    for item in files:
        name = item["path"]
        rel = safe_rel(name)
        expected_bytes = item["bytes"]
        expected_sha = item["sha256"]
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError("Invalid remote artifact size")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hash_file(target) == expected_sha:
            continue
        if target.exists():
            raise FileExistsError("Refusing to replace mismatched artifact: " + str(target))
        part = target.with_name(target.name + ".part")
        with part.open("wb") as stream:
            for offset in range(0, expected_bytes, CHUNK_MAX):
                length = min(CHUNK_MAX, expected_bytes - offset)
                result = remote(dseq, ["chunk", "--path", name, "--offset",
                                       str(offset), "--length", str(length)], context)
                data = base64.b64decode(result["data"], validate=True)
                if len(data) != length:
                    raise IOError("Incomplete private artifact chunk")
                stream.write(data)
        if part.stat().st_size != expected_bytes or hash_file(part) != expected_sha:
            part.unlink(missing_ok=True)
            raise IOError("Artifact integrity mismatch: " + name)
        os.replace(part, target)
    atomic_json(dest / "verified_manifest.json", listing)
    return len(files)


def is_closed(value):
    if isinstance(value, dict):
        if any(str(v).lower() == "closed" for k, v in value.items()
               if k in ("state", "status")):
            return True
        return any(is_closed(v) for v in value.values() if isinstance(v, (dict, list)))
    if isinstance(value, list):
        return any(is_closed(v) for v in value)
    return False


def close(state, context):
    if state.get("state") == "closed":
        return True
    if not state.get("dseq"):
        return False
    dseq = state["dseq"]
    state_file = Path(state["out"]) / "state.json"
    state["state"] = "closing"
    atomic_json(state_file, state)
    try:
        cli(["close", str(dseq), "--yes", "-o", "json"], context=context)
    except subprocess.CalledProcessError:
        # Already closed is an expected recovery case. Verify before accepting it.
        pass
    try:
        data = json.loads(cli(["console", "deployment", "get", str(dseq),
                               "-o", "json"], context=context))
        verified = is_closed(data)
    except Exception:
        verified = False
    state["state"] = "closed" if verified else "closure_unverified"
    atomic_json(state_file, state)
    return verified


def await_and_export(state, context, poll_seconds, max_bytes):
    deadline = state["created_at"] + state["max_hours"] * 3600
    failures = 0
    while time.time() < deadline:
        try:
            result = remote(state["dseq"], ["status"], context)
            failures = 0
        except Exception:
            failures += 1
            if failures >= 4:
                raise
            time.sleep(poll_seconds)
            continue
        if result.get("status") in ("completed", "failed"):
            state["worker_status"] = result
            atomic_json(Path(state["out"]) / "state.json", state)
            state["downloaded_files"] = download(state, context, max_bytes)
            atomic_json(Path(state["out"]) / "state.json", state)
            if result["status"] != "completed":
                raise RuntimeError("Remote training failed; recovered available artifacts")
            return
        time.sleep(poll_seconds)
    raise TimeoutError("Campaign exceeded approved lease lifetime")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="experiments/gpu-smoke.json")
        p.add_argument("--image", required=True, help="GHCR @sha256:<digest>")
        p.add_argument("--out", default=str(BASE))
        p.add_argument("--max-runs", type=int, default=1)
        p.add_argument("--hours", type=int, default=2)
        p.add_argument("--context")
        p.add_argument("--max-artifact-mib", type=int, default=256)
        p.add_argument("--poll-seconds", type=int, default=20)
        if name == "run":
            p.add_argument("--approve-paid-deployment", action="store_true")
    for name in ("recover", "close"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--context")
        p.add_argument("--poll-seconds", type=int, default=20)
        p.add_argument("--max-artifact-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1 or args.max_artifact_mib < 1:
        parser.error("Invalid poll interval / artifact budget")
    if args.command in ("plan", "run"):
        sdl, state = prepare(args)
        cli(["sdl", "validate", str(sdl)], context=args.context)
        cli(["deploy", str(sdl), "--dry-run"], context=args.context)
        if args.command == "plan":
            print(json.dumps(state, indent=2))
            return
        if not args.approve_paid_deployment:
            parser.error("Paid deployment requires --approve-paid-deployment")
        if (Path(args.out) / "state.json").exists():
            parser.error("Existing deployment record; use recover or choose another --out")
        console_check(args.context)
        state["created_at"] = time.time()
        state["state"] = "deploying"
        atomic_json(Path(args.out) / "state.json", state)
        try:
            dseq = deploy(sdl, state, Path(args.out) / "state.json", args.context)
            # Console-side TTL protects against a sleeping laptop.
            cli(["console", "deployment", "settings", dseq, str(args.hours)],
                context=args.context)
            state["state"] = "running"
            atomic_json(Path(args.out) / "state.json", state)
            await_and_export(state, args.context, args.poll_seconds,
                             args.max_artifact_mib * 1024 * 1024)
        finally:
            if not close(state, args.context):
                print("CLOSURE NOT VERIFIED. Check: akt console deployment list active",
                      file=sys.stderr)
    else:
        state_file = Path(args.out) / "state.json"
        state = json.loads(state_file.read_text())
        if not state.get("dseq"):
            parser.error("Missing deployment ID; inspect akt console deployment list active")
        console_check(args.context)
        try:
            if args.command == "recover" and state["state"] != "closed":
                await_and_export(state, args.context, args.poll_seconds,
                                 args.max_artifact_mib * 1024 * 1024)
        finally:
            if not close(state, args.context):
                print("CLOSURE NOT VERIFIED. Inspect Akash Console.", file=sys.stderr)
    print("Local run folder:", args.out)


if __name__ == "__main__":
    main()
