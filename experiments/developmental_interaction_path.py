"""Small online developmental path from waves to interactive RGB prediction.

This is deliberately an experience learner, not a second backprop trainer.  A
retinal RGB sheet is reduced into fast/slow local activity, a phase field makes
traveling waves, and a local Hebbian rule refines the connections used by the
readout from prediction surprise.  The same loop accepts synthetic 3D views or
webcam frames; depth, object identity and pose never enter the learner.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src import physical_fields, world3d


def _pool(frame, size):
    """Average RGB retinal activity into a fixed small sheet."""
    h, w, c = frame.shape
    if c != 3 or h % size or w % size:
        raise ValueError("RGB frame must be divisible by sheet size")
    return frame.reshape(size, h // size, size, w // size, 3).mean((1, 3))


class DevelopmentalSheet:
    """Online sheet with phase, surprise trace, and plastic local connections."""

    def __init__(self, sheet=8, channels=7, seed=11, plasticity=.04):
        if sheet < 2 or channels < 3 or not 0 <= plasticity <= .2:
            raise ValueError("Invalid sheet, channel or plasticity setting")
        self.sheet, self.channels = sheet, channels
        self.rate = float(plasticity)
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(-np.pi, np.pi, (sheet, sheet)).astype(np.float32)
        self.hidden = np.zeros((sheet, sheet, channels), np.float32)
        self.trace = np.zeros((sheet, sheet, channels), np.float32)
        self.homeostasis = np.zeros((channels,), np.float32)
        # Connections are experience-dependent: rows are presynaptic features,
        # columns are RGB prediction channels.  Start near zero so learning is
        # carried by observations rather than a pre-trained visual kernel.
        self.connections = .01 * rng.standard_normal(
            (sheet, sheet, channels, 3)).astype(np.float32)
        self.previous = np.zeros((sheet, sheet, 3), np.float32)
        self.forecast = np.zeros((sheet, sheet, 3), np.float32)

    def _features(self, frame):
        fast = _pool(frame, self.sheet)
        slow = fast.reshape(self.sheet // 2, 2, self.sheet // 2, 2, 3).mean((1, 3))
        slow = np.repeat(np.repeat(slow, 2, 0), 2, 1)
        # Phase is a non-parametric traveling-wave carrier; it modulates the
        # local sensory representation without supplying hidden world state.
        wave = np.sin(self.phase)[..., None].astype(np.float32)
        return np.concatenate((fast, slow, wave), axis=-1)[..., :self.channels]

    def observe(self, frame, action=(0., 0.)):
        """Absorb one retinal observation and return the next-frame forecast."""
        x = self._features(frame)
        retinal = _pool(frame, self.sheet)
        surprise = np.mean(np.abs(retinal - self.forecast))
        action_signal = .02 * float(action[0]) + .01 * float(action[1])
        self.trace = .82 * self.trace + .18 * np.tanh(x * surprise)
        pre = np.tanh(x + .25 * self.hidden + .08 * self.trace +
                      self.homeostasis + action_signal)
        post = np.tanh(pre + .05 * np.roll(pre, 1, axis=0) + .05 * np.roll(pre, 1, axis=1))
        # Activity-dependent refinement, normalized to avoid runaway weights.
        correlation = pre[..., :, None] * post[..., None, :3]
        correlation = correlation - correlation.mean(axis=2, keepdims=True)
        self.connections += self.rate * float(np.clip(surprise, 0., 1.)) * correlation
        self.connections *= np.float32(1. - self.rate * .02)
        activity = np.mean(np.abs(post), axis=(0, 1))
        self.homeostasis += .02 * (.12 - activity)
        self.homeostasis = np.clip(self.homeostasis, -.25, .25)
        norms = np.linalg.norm(self.connections, axis=2, keepdims=True)
        self.connections /= np.maximum(1., norms)
        self.hidden = post
        self.phase += .25 + .08 * np.roll(self.phase, 1, axis=0) - .08 * self.phase
        self.phase += .01 * float(action[0]) + .005 * float(action[1])
        delta = np.tanh(np.einsum("hwc,hwcr->hwr", self.hidden, self.connections))
        prediction = np.clip(self.previous + .18 * delta, 0., 1.)
        self.previous = retinal
        self.forecast = prediction
        return prediction

    def metrics(self):
        coherence = abs(np.mean(np.exp(1j * self.phase)))
        return {"phase_coherence": float(coherence),
                "connection_l1": float(np.mean(np.abs(self.connections))),
                "trace_l1": float(np.mean(np.abs(self.trace))),
                "homeostatic_drive_l1": float(np.mean(np.abs(self.homeostasis)))}


def _retinal_frames(source, seed, frames, size):
    if source == "world3d":
        record = world3d.episode(seed, frames, size, "train")
        return record["rgb"], {"source": source, "actions": record["camera_actions"].tolist(),
                               "depth_seen_by_model": False}
    if source == "wave":
        record = physical_fields.episode(seed, "wave", size, frames)
        values = record["observed"]
        rgb = np.repeat(values[..., None], 3, axis=-1)
        rgb = (rgb - rgb.min()) / max(1e-6, float(rgb.max() - rgb.min()))
        return rgb.astype(np.float32), {"source": source, "actions": [],
                                       "depth_seen_by_model": False}
    if source == "webcam":
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("webcam source requires opencv-python") from error
        capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError("Cannot open RGB webcam 0")
        captured = []
        try:
            for _ in range(frames):
                ok, raw = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
                captured.append(rgb.astype(np.float32) / 255.)
        finally:
            capture.release()
        if len(captured) < 3:
            raise RuntimeError("Webcam produced fewer than three RGB frames")
        return np.stack(captured), {"source": source, "actions": [],
                                   "depth_seen_by_model": False}
    raise ValueError("source must be world3d, wave or webcam")


def run(out="runs/developmental-interaction-path", seed=11, frames=24,
        size=32, sheet=8, plasticity=.04, source="world3d"):
    if frames < 3 or size < 16 or size % sheet or sheet % 2:
        raise ValueError("frames >=3, size divisible by an even sheet required")
    observations, provenance = _retinal_frames(source, seed, frames, size)
    learner = DevelopmentalSheet(sheet, 7, seed, plasticity)
    errors = []
    for index, frame in enumerate(observations):
        prediction = learner.observe(frame, provenance.get("actions", [])[index-1]
                                     if index and index-1 < len(provenance.get("actions", [])) else (0., 0.))
        # Compare only against the current observation after issuing the prior
        # forecast; this is the same temporal discipline used by webcam scoring.
        errors.append(float(np.mean((prediction - _pool(frame, sheet)) ** 2)))
    result = {"experiment": "developmental_interaction_path_v1",
              "frames": frames, "retinal_size": [size, size, 3],
              "sheet_size": [sheet, sheet], "source": source,
              "plasticity": plasticity, "online_updates": frames,
              "mean_prediction_mse": float(np.mean(errors[1:])),
              "first_prediction_mse": errors[0], "last_prediction_mse": errors[-1],
              "learning_signal": float(errors[0] - errors[-1]),
              "learner": learner.metrics(), "provenance": provenance,
              "limitations": "CPU smoke path; no depth, pose, identity or future frame enters the learner. "
                             "For webcam use, feed preprocessed RGB frames from the existing live capture path."}
    dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/developmental-interaction-path")
    p.add_argument("--source", choices=("world3d", "wave", "webcam"), default="world3d")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--sheet", type=int, default=8)
    p.add_argument("--plasticity", type=float, default=.04)
    a = p.parse_args()
    print(json.dumps(run(a.out, a.seed, a.frames, a.size, a.sheet,
                         a.plasticity, a.source), indent=2))


if __name__ == "__main__":
    main()
