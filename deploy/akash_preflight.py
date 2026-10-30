"""Render a GPU smoke SDL only from a verified immutable image reference."""

import argparse
import os
import re
from pathlib import Path


IMAGE = re.compile(
    r"ghcr\.io/walmsley-lab/ml-innovation-18-predictive-mirror@sha256:[a-fA-F0-9]{64}\Z"
)


def replace_once(document, name, pattern, replacement):
    """Replace one required SDL field, refusing ambiguous or changed templates."""
    result, count = re.subn(pattern, replacement, document, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one {name} field; found {count}")
    return result


def render(template, image, steps=3, batch_size=2):
    """Generate a bounded, GPU-required smoke configuration without deploying."""
    if not IMAGE.fullmatch(image):
        raise ValueError("Image must be this repo's immutable GHCR @sha256: digest")
    if not (1 <= steps <= 3 and 1 <= batch_size <= 2):
        raise ValueError("Smoke limits: 1..3 training steps, batch size 1..2")
    if not re.search(r"^\s+denom:\s*uact(?:\s*(?:#.*)?)?$", template, re.MULTILINE):
        raise ValueError("Expected uact billing denomination")
    price = re.search(r"^\s+amount:\s*(\d+)\s*(?:#.*)?$", template, re.MULTILINE)
    if not price or int(price.group(1)) > 2500:
        raise ValueError("Maximum provider bid must not exceed 2500 uact/block")
    if not re.search(r"^\s+gpu:\s*\n\s+units:\s*1\s*$", template, re.MULTILINE):
        raise ValueError("Smoke configuration must request exactly one GPU")
    if not re.search(r"^\s+-\s+port:\s*8080\s*$", template, re.MULTILINE):
        raise ValueError("Expected synthetic artifact endpoint on port 8080")
    result = replace_once(template, "image", r"^(\s+image:\s*)\S+(?:\s+#.*)?$",
                          lambda match: match.group(1) + image)
    replacements = {
        "RUN_MODE": "train",
        "STEPS": str(steps),
        "BATCH_SIZE": str(batch_size),
        "CHECKPOINT_EVERY": "1",
        "LOSS_MODE": "plain",
        "REQUIRE_GPU": "1",
        "RESUME": "0",
    }
    # RESUME is optional in the current template; required once PR #11 lands.
    for name, value in replacements.items():
        pattern = rf"^(\s+-\s+{name}=).*$"
        matches = len(re.findall(pattern, result, re.MULTILINE))
        if name == "RESUME" and matches == 0:
            continue
        result = replace_once(result, name, pattern,
                              lambda match, val=value: match.group(1) + val)
    return result


def save(output, document):
    """Do not silently replace a previously reviewed deployment manifest."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text() != document:
            raise FileExistsError(f"Refusing to overwrite different SDL: {output}")
        return output
    temporary = output.with_name(output.name + ".part")
    try:
        with temporary.open("x") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True,
                        help="Full ghcr.io/...@sha256:<64hex> verified digest")
    parser.add_argument("--template", default="deploy/akash-gpu.yaml")
    parser.add_argument("--out", default="runs/akash-smoke.yaml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    document = render(Path(args.template).read_text(),
                      args.image, args.steps, args.batch_size)
    path = save(args.out, document)
    print(f"Prepared {path}; no Akash deployment created.")
    print(f"Next: akt sdl validate {path}")
    print(f"Then: akt deploy {path} --dry-run")
    print(f"Paid launch, only after checking bids: akt deploy {path}")


if __name__ == "__main__":
    main()
