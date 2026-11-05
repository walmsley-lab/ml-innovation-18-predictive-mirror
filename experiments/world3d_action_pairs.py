"""Measure whether a frozen RGB forecaster is sensitive to actual 3D actions."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import world3d_action_pairs as pairs, action_rgb_nca as nca


def load(checkpoint=None,untrained_smoke=False):
    if checkpoint is None:
        if not untrained_smoke:
            return None,None
        return nca.init(jax.random.PRNGKey(17),4),"UNTRAINED_SMOKE"
    with np.load(checkpoint,allow_pickle=False) as f:
        params={key:jnp.asarray(f[key]) for key in f.files}
    return params,hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()


def run(checkpoint=None,out="runs/world3d-action-pairs/metrics.json",
        seeds=(11,23,41),context=5,horizon=5,size=24,
        untrained_smoke=False):
    if not seeds:
        raise ValueError("Need world seeds")
    params,sha=load(checkpoint,untrained_smoke)
    reports=[]
    for seed in seeds:
        positive,negative=pairs.pair(seed,context,horizon,size)
        if params is None:
            scores=pairs.scores(positive,negative,context=context)
        else:
            common=jnp.asarray(positive["rgb"][:context][None])
            planned_pos=jnp.asarray(positive["camera_actions"][None])
            planned_neg=jnp.asarray(negative["camera_actions"][None])
            forecast_p=np.asarray(nca.rollout(
                params,common,planned_pos,horizon,True))[0]
            forecast_n=np.asarray(nca.rollout(
                params,common,planned_neg,horizon,True))[0]
            scores=pairs.scores(positive,negative,forecast_p,
                               forecast_n,context)
        reports.append({"seed":int(seed),"scores":scores})
    report={"protocol":"paired-3d-counterfactual-action-v1",
            "seeds":list(map(int,seeds)),"context":context,"horizon":horizon,
            "size":size,"checkpoint_sha256":sha,"untrained_smoke":untrained_smoke,
            "results":reports,
            "limits":"Model sees identical RGB past and only known planned "
                     "camera commands. Target contrast is evaluator-only. "
                     "Untrained smoke is not evidence of sensitivity after learning. "
                     "No geometric understanding is established by action response."}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--out",default="runs/world3d-action-pairs/metrics.json")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    p.add_argument("--context",type=int,default=5)
    p.add_argument("--horizon",type=int,default=5)
    p.add_argument("--size",type=int,default=24)
    a=p.parse_args()
    print(json.dumps(run(a.checkpoint,a.out,a.seeds,a.context,a.horizon,
                         a.size,a.untrained_smoke)["results"],indent=2))


if __name__=="__main__":
    main()
