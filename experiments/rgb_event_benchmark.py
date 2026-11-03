"""Run frozen RGB models and observation-only baselines on identical world episodes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src import rgb_benchmark, rgb_event_metrics, rgb_motion_world


def run(output="runs/rgb-event-benchmark/metrics.json", checkpoint=None,
        seed=81000, episodes=16, context=8, horizon=10, size=32,
        balls=2, occlusion=True, camera_motion=False, ticks=2):
    if episodes < 1 or context < 2 or horizon < 1 or balls < 1 or balls > 4:
        raise ValueError("Invalid experiment dimensions")
    records = [rgb_motion_world.episode(
        seed+i, steps=context+horizon, size=size, balls=balls,
        occlusion=occlusion, camera_motion=camera_motion)
        for i in range(episodes)]
    frames = rgb_benchmark.frames(records)
    context_frames = frames[:, :context]
    predicts = rgb_benchmark.controls(context_frames, horizon)
    provenance = {"checkpoint": None, "checkpoint_sha256": None}
    if checkpoint is not None:
        import jax.numpy as jnp
        from src import predictive_rgb_nca as nca
        with np.load(checkpoint, allow_pickle=False) as archive:
            params = {k: jnp.asarray(archive[k]) for k in archive.files}
        predicts["nca"] = np.asarray(nca.rollout(
            params, jnp.asarray(context_frames), horizon, ticks))
        provenance = {"checkpoint": str(checkpoint),
                      "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
    leads = tuple(sorted({horizon, *(x for x in (1,3,5,10) if x <= horizon)}))
    result = {
        "protocol": "rgb-event-evaluator-v1",
        "scenario": {"seed_start": seed, "episodes": episodes, "balls": balls,
                     "occlusion": occlusion, "camera_motion": camera_motion,
                     "context": context, "horizon": horizon, "size": size},
        "model": provenance,
        "metrics": {name: rgb_event_metrics.score(values, records, context, leads)
                    for name, values in predicts.items()},
        "limits": "Palette-based localization is a diagnostic, not an object detector. "
                  "Return-after-invisibility includes off-screen re-entry; it is NOT "
                  "a pure occlusion metric. Scenario labels and simulator truth "
                  "are evaluator-only. Seed exclusion relative to any checkpoint "
                  "training/selection is the experimenter's responsibility.",
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--out", default="runs/rgb-event-benchmark/metrics.json")
    p.add_argument("--seed", type=int, default=81000)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--balls", type=int, default=2)
    p.add_argument("--no-occlusion", action="store_true")
    p.add_argument("--camera-motion", action="store_true")
    a = p.parse_args()
    report = run(a.out, a.checkpoint, a.seed, a.episodes, a.context,
                 a.horizon, a.size, a.balls, not a.no_occlusion, a.camera_motion)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
