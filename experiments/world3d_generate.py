"""Generate 3D perspective frames and evaluator-only geometric truth."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import world3d


def run(out="runs/world3d-smoke",seed=1001,episodes=4,steps=12,
        size=32,domain="train",backend="analytic"):
    if episodes<1:
        raise ValueError("Need at least one episode")
    path=Path(out)
    path.mkdir(parents=True,exist_ok=True)
    if backend=="mujoco":
        from src.world3d_mujoco import smoke
        rgb=smoke(size=size,steps=steps)
        np.savez_compressed(path/"mujoco-rgb.npz",rgb=rgb)
        report={"backend":"mujoco","frames":len(rgb),
                "limits":"RGB-only smoke. Does not establish calibrated depth or segmentation."}
    elif backend=="analytic":
        records=[world3d.episode(seed+i,steps,size,domain) for i in range(episodes)]
        arrays=("rgb","depth","instance_id","camera_actions","K",
                "T_world_camera","centers_world","times_s")
        np.savez_compressed(path/"episodes.npz",
            **{key:np.stack([x[key] for x in records]) for key in arrays})
        report={"backend":"ray-sphere-analytic-v1","episodes":episodes,
                "seed_root":seed,"steps":steps,"size":size,"domain":domain,
                "model_inputs":["rgb","camera_actions_if_planned"],
                "evaluation_only":["depth","instance_id","K","T_world_camera",
                                   "centers_world"],
                "units":"meters_xyz_seconds",
                "limits":"Only rigid spheres and prescribed camera motion; "
                         "not photorealism or arbitrary physics."}
    else:
        raise ValueError("Unknown world backend")
    (path/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/world3d-smoke")
    p.add_argument("--seed",type=int,default=1001)
    p.add_argument("--episodes",type=int,default=4)
    p.add_argument("--steps",type=int,default=12)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--domain",choices=world3d.DOMAINS,default="train")
    p.add_argument("--backend",choices=("analytic","mujoco"),default="analytic")
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seed,a.episodes,a.steps,a.size,
                         a.domain,a.backend),indent=2))


if __name__=="__main__":
    main()
