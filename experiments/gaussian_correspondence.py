"""Frozen Gaussian slot diagnostic on motion, partial occlusion and views."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
from src import world3d, rgb_gaussian_nca as gaussian, gaussian_correspondence as evaluation
from experiments.gaussian_active_view import load


def run(out="runs/gaussian-correspondence/metrics.json",checkpoint=None,
        untrained_smoke=False,seeds=(11,23,41),steps=10,size=32,tolerance=4.):
    if not seeds or steps<5 or (not checkpoint and not untrained_smoke):
        raise ValueError("Require frozen checkpoint or explicit smoke")
    params=(load(checkpoint) if checkpoint else
            gaussian.init(jax.random.PRNGKey(19),3,8))
    rows=[]
    for domain in ("train","ood_camera","controlled_partial_occlusion"):
        for seed in seeds:
            rec=(evaluation.partial_episode(int(seed),steps,size) if
                 domain=="controlled_partial_occlusion" else
                 world3d.episode(int(seed),steps,size,domain))
            row=evaluation.trace(params,rec,tolerance)
            row.update({"domain":domain,"seed":int(seed)})
            rows.append(row)
    manifest={"protocol":"permutation-aware-gaussian-correspondence-v1",
              "checkpoint_sha256":(hashlib.sha256(
                  Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None),
              "untrained_smoke":untrained_smoke,"seeds":list(map(int,seeds)),
              "rows":rows,"tolerance_px":tolerance,
              "limits":"No frame-to-frame oracle correspondence is given "
                  "to inferred slots. Evaluator matches independently "
                  "visible ground-truth objects per frame. Partial and "
                  "fully hidden episodes are not equivalent to guaranteed "
                  "object reappearance tests."}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/gaussian-correspondence/metrics.json")
    p.add_argument("--checkpoint")
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=10)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--tolerance",type=float,default=4.)
    a=p.parse_args()
    print(json.dumps({"evaluated_sequences":len(run(
        a.out,a.checkpoint,a.untrained_smoke,a.seeds,a.steps,a.size,
        a.tolerance)["rows"])},indent=2))


if __name__=="__main__":
    main()
