"""Freeze perspective and depth tests on independent procedural worlds."""
import argparse
import json
from pathlib import Path
from src import world3d,world3d_benchmark as bench


def run(out="runs/world3d-benchmark/metrics.json",seed=82000,
        episodes=16,context=8,horizon=10,size=32,domain="ood_camera"):
    if episodes<1 or context<2 or horizon<1:
        raise ValueError("Invalid frozen 3D evaluation")
    records=[world3d.episode(seed+i,context+horizon,size,domain)
             for i in range(episodes)]
    baselines=bench.diagnostic_baselines(records,context,horizon)
    leads=tuple(sorted({horizon,*(x for x in (1,3,5,10) if x<=horizon)}))
    result={"protocol":"frozen-3d-perspective-v1",
            "generator":"ray-sphere-analytic-v1",
            "test_seed_root":seed,"episodes":episodes,"context":context,
            "horizon":horizon,"domain":domain,
            "scores":{k:bench.score(records,v,context,leads)
                      for k,v in baselines.items()},
            "controls":{
                "rgb_persistence":{"privileged":False},
                "oracle_depth_reprojection":{"privileged":True,
                    "reason":"Uses true prior depth and future camera poses; "
                    "evaluation-only geometry diagnostic, not RGB competitor"}},
            "limitations":"Data are analytic spheres, not photorealistic. "
                "Forward reprojection leaves disocclusion holes and ignores "
                "object motion. Different test domains must be frozen before "
                "checkpoint selection."}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/world3d-benchmark/metrics.json")
    p.add_argument("--seed",type=int,default=82000)
    p.add_argument("--episodes",type=int,default=16)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--domain",choices=world3d.DOMAINS,default="ood_camera")
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seed,a.episodes,a.context,a.horizon,a.size,
                         a.domain)["scores"],indent=2))


if __name__=="__main__":
    main()
