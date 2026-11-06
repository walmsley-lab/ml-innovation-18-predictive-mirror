"""Small CPU-only paired-observation generator for audiovisual geometry studies.

These are hand-authored synthetic scenes, not decoded thoughts or perception data.
Simulator fields and control labels are evaluation-only; no future observations
are passed to a forecaster by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODES = ("spiral", "fold", "lattice", "chamber")


def scene(seed, mode="spiral", light=0.15, detail=0.55, flow=0.42,
          pitch=0.42, coupling=0.65):
    if mode not in MODES or not 0 <= int(seed) <= 999999:
        raise ValueError("Invalid mode or seed")
    values = (light, detail, flow, pitch, coupling)
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Scene controls must be finite values in [0, 1]")
    return dict(seed=int(seed), mode=mode, light=float(light),
                detail=float(detail), flow=float(flow),
                pitch=float(pitch), coupling=float(coupling))


def field(size, seconds, parameters):
    """Returns the underlying intensity field; independent of illumination."""
    if size < 8 or not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Expected size >= 8 and nonnegative finite time")
    s = parameters
    coordinates = (2 * (np.arange(size) + 0.5) / size) - 1
    x, y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    t = seconds * (0.14 + 0.64 * s["flow"])
    n = s["seed"] * 127.1 + 39.7
    raw = np.sin(n) * 43758.5453
    seed_phase = (raw - np.floor(raw)) * 2 * np.pi
    density = 5 + int(np.floor(s["detail"] * 11 + 0.5))
    pitch_phase = (1.2 + 7 * s["pitch"]) * s["coupling"]
    theta = angle + 0.23 * np.sin(radius * 3 - t + seed_phase)
    if s["mode"] == "fold":
        a = np.cos(density * theta + 2.6 * np.sin(4 * radius - t))
        b = np.cos(16 * radius - 1.5 * t + pitch_phase * theta)
    elif s["mode"] == "lattice":
        a = np.cos(density * (x + 0.25 * np.sin(y * 5 + t)))
        b = np.cos(density * (y - 0.25 * np.sin(x * 5 - t)) + pitch_phase)
    elif s["mode"] == "chamber":
        a = np.cos(density * np.log(radius + 0.12) - t + seed_phase)
        b = np.cos((density / 2) * theta + 3 * np.sin(3 * radius - t))
    else:
        a = np.sin(density * theta / 2 + 7 * np.log(radius + 0.10) - t)
        b = np.cos(19 * radius - 1.3 * t + pitch_phase * np.sin(theta * 3))
    interference = np.sin(4 * radius + 2 * theta - 0.7 * t + seed_phase)
    intensity = np.clip(0.5 + 0.26 * a + 0.19 * b + 0.05 * interference, 0, 1)
    hue = 0.5 + 0.5 * np.cos(theta * 2 + radius * 4 - 0.3 * t + seed_phase)
    return intensity, hue


def render(intensity, hue, light):
    """Display transform: identical hidden fields can yield different RGB images."""
    if not np.isfinite(light) or not 0 <= light <= 1:
        raise ValueError("Light must be in [0, 1]")
    accents = np.stack([
        0.5 + 0.5 * np.cos(2 * np.pi * hue + offset)
        for offset in (0, 2.09, 4.18)
    ], axis=-1)
    strength = (0.13 + 0.87 * intensity) * (0.35 + 0.65 * np.abs(2 * intensity - 1))
    strength = strength[..., None] * (0.32 + 0.68 * accents)
    base = 10 + light * 234
    pixels = (base + (1-light)*195*strength if light < 0.5
              else base - light*172*strength)
    return np.floor(np.clip(pixels, 0, 255) + 0.5).astype(np.uint8)


def episode(seed, mode, frames=16, size=32, dark=0.10, bright=0.90):
    """Paired light/dark videos and an audio cue driven by a shared control."""
    if frames < 2 or size < 8 or not (0 <= dark < bright <= 1):
        raise ValueError("Invalid episode dimensions or lighting pair")
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0, 2*np.pi))
    params = scene(seed, mode=mode, coupling=0.65)
    fields, dark_rgb, bright_rgb, tones = [], [], [], []
    for index in range(frames):
        seconds = index / 10.0  # synthetic timeline, not a wall-clock claim
        pitch = 0.5 + 0.28 * np.sin(0.7 * seconds + phase)
        s = {**params, "pitch": float(pitch)}
        intensity, hue = field(size, seconds, s)
        fields.append(intensity.astype(np.float32))
        dark_rgb.append(render(intensity, hue, dark))
        bright_rgb.append(render(intensity, hue, bright))
        tones.append(110 * 2 ** (1.5 * pitch))
    return (np.stack(dark_rgb), np.stack(bright_rgb),
            np.stack(fields), np.asarray(tones, dtype=np.float32))


def generate(seed=41000, episodes=8, frames=16, size=32, dark=0.10, bright=0.90):
    if episodes < 1 or seed < 0 or seed + episodes - 1 > 999999:
        raise ValueError("Invalid seed or episode count")
    sequences = [episode(seed+i, MODES[i % len(MODES)], frames, size, dark, bright)
                 for i in range(episodes)]
    dark_rgb, bright_rgb, underlying, cue_hz = [
        np.stack([sequence[j] for sequence in sequences]) for j in range(4)]
    return dict(frames_dark=dark_rgb, frames_bright=bright_rgb,
                latent_intensity=underlying, cue_hz=cue_hz)


def run(out="runs/perceptual-atlas/dataset.npz", seed=41000, episodes=8,
        frames=16, size=32, dark=0.10, bright=0.90):
    data = generate(seed, episodes, frames, size, dark, bright)
    output = Path(out)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Use a fresh output path; files are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    manifest = dict(protocol="perceptual-atlas-synthetic-v1", seed=seed,
                    episodes=episodes, frames=frames, size=size,
                    dark=dark, bright=bright, modes=[MODES[i % len(MODES)]
                    for i in range(episodes)], shape=list(data["frames_dark"].shape),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    inputs="RGB observations and optional cue_hz per frame",
                    evaluator_only="latent_intensity, generator parameters, future frames",
                    interpretation="Controlled synthetic sketches, not inferred mental states. "
                    "Matching underlying fields do not establish learned invariance.")
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/perceptual-atlas/dataset.npz")
    p.add_argument("--seed", type=int, default=41000)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--dark", type=float, default=0.10)
    p.add_argument("--bright", type=float, default=0.90)
    a = p.parse_args()
    print(json.dumps(run(a.out, a.seed, a.episodes, a.frames,
                         a.size, a.dark, a.bright), indent=2))


if __name__ == "__main__":
    main()
