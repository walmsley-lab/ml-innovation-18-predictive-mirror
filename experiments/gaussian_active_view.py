"""Evaluate trained RGB Gaussian ensemble under paired camera policies."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_gaussian_nca as model, gaussian_active_view as active


def load(path):
    with np.load(path,allow_pickle=False) as f:
        items={key:jnp.asarray(f[key]) for key in f.files}
    return {
        "dynamics":{k.removeprefix("dynamics_"):v for k,v in items.items()
                    if k.startswith("dynamics_")},
        **{k:v for k,v in items.items() if not k.startswith("dynamics_")},
    }


def run(checkpoints=(),out="runs/gaussian-active-view/metrics.json",
        seeds=(11,23,41),domains=("ood_depth","ood_camera"),
        steps=5,size=24,untrained_smoke=False):
    if untrained_smoke:
        models=[model.init(jax.random.PRNGKey(k),3,4) for k in (1,2)]
        digests=["UNTRAINED_SMOKE"]*len(models)
    else:
        if len(checkpoints)<2:
            raise ValueError("At least two frozen trained checkpoints required")
        models=[load(path) for path in checkpoints]
        digests=[hashlib.sha256(Path(path).read_bytes()).hexdigest()
                 for path in checkpoints]
    report=active.compare(models,seeds,domains,steps,size)
    report["model_checkpoint_sha256"]=digests
    report["untrained_smoke"]=bool(untrained_smoke)
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints",nargs="*",default=[])
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--out",default="runs/gaussian-active-view/metrics.json")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    p.add_argument("--domains",nargs="+",choices=("train",)+
                   ("ood_depth","ood_camera","ood_appearance"),
                   default=["ood_depth","ood_camera"])
    p.add_argument("--steps",type=int,default=5)
    p.add_argument("--size",type=int,default=24)
    a=p.parse_args()
    result=run(a.checkpoints,a.out,a.seeds,a.domains,a.steps,a.size,
               a.untrained_smoke)
    print(json.dumps({d:x["mean_home_mse"]
                      for d,x in result["domains"].items()},indent=2))


if __name__=="__main__":
    main()
