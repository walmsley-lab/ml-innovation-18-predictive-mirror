"""Evaluate active versus fixed/random gaze on identical hidden 3D worlds."""
import argparse
import hashlib
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from src import active_world3d,action_rgb_nca


def run(checkpoints=(),out="runs/active-world3d/metrics.json",
        seeds=(11,23,37),steps=5,size=24,untrained_smoke=False):
    if not checkpoints and not untrained_smoke:
        raise ValueError("Provide frozen trained checkpoints or --untrained-smoke")
    provenance=[]
    if untrained_smoke:
        import jax
        ensemble=[action_rgb_nca.init(jax.random.PRNGKey(i),4) for i in (1,2)]
        provenance=["UNTRAINED_SMOKE"]*len(ensemble)
    else:
        ensemble=[]
        for path in checkpoints:
            with np.load(path,allow_pickle=False) as file:
                ensemble.append({k:jnp.asarray(file[k]) for k in file.files})
            provenance.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    result=active_world3d.compare(ensemble,seeds,steps,size)
    result["ensemble_sha256"]=provenance
    result["untrained_smoke"]=bool(untrained_smoke)
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints",nargs="*",default=[])
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--out",default="runs/active-world3d/metrics.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,37])
    p.add_argument("--steps",type=int,default=5)
    p.add_argument("--size",type=int,default=24)
    a=p.parse_args()
    print(json.dumps(run(a.checkpoints,a.out,a.seeds,a.steps,a.size,
                         a.untrained_smoke)["paired"],indent=2))


if __name__=="__main__":
    main()
