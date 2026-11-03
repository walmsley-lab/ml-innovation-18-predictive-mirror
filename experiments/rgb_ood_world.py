"""Generate a deterministic family of harder RGB worlds without training a model."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import rgb_robust_world as world


def run(path="runs/rgb-ood-world/manifest.json", episodes=4, steps=18, size=32,
        domain="ood_combined", seed=90000):
    if episodes < 1:
        raise ValueError("Need episodes")
    scenes = [world.episode(seed+i, steps=steps, size=size, domain=domain)
              for i in range(episodes)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.parent/"examples.npz",
        frames=np.stack([s["frames"] for s in scenes]),
        camera_actions=np.stack([s["camera_actions"] for s in scenes]))
    manifest = {
        "generator": "rgb_robust_world_v1", "domain": domain, "seed_start": seed,
        "episodes": episodes, "steps": steps, "size": size,
        "rgb_path": "examples.npz",
        "model_visible_by_default": ["frames"],
        "action_conditioned_only": ["camera_actions"],
        "evaluator_only": ["positions", "velocities", "visible", "in_view",
                           "occluded", "colors", "radii", "camera_xy"],
        "caveat": "This is 2D accelerated-disc simulation, NOT camera perspective "
                  "or learned 3D understanding. Future randomly commanded camera "
                  "actions are unknowable unless explicitly supplied.",
    }
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/rgb-ood-world/manifest.json")
    parser.add_argument("--domain", choices=world.DOMAINS, default="ood_combined")
    parser.add_argument("--seed",type=int,default=90000)
    parser.add_argument("--episodes",type=int,default=4)
    parser.add_argument("--steps",type=int,default=18)
    parser.add_argument("--size",type=int,default=32)
    a = parser.parse_args()
    print(json.dumps(run(a.out,a.episodes,a.steps,a.size,a.domain,a.seed),indent=2))


if __name__ == "__main__":
    main()
