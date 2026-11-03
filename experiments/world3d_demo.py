"""Open a self-contained 3D forecast storyboard and active-policy evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import action_rgb_nca as model,world3d,world3d_benchmark as bench
from src import active_world3d,world3d_storyboard
from experiments.active_world3d import run as load_active


def load(checkpoints,untrained_smoke):
    if untrained_smoke:
        return ([model.init(jax.random.PRNGKey(i),4) for i in (1,2)],
                ["UNTRAINED_SMOKE","UNTRAINED_SMOKE"])
    if not checkpoints:
        raise ValueError("Need a trained checkpoint or --untrained-smoke")
    ensemble,digests=[],[]
    for file in checkpoints:
        with np.load(file,allow_pickle=False) as weights:
            ensemble.append({key:jnp.asarray(weights[key]) for key in weights.files})
        digests.append(hashlib.sha256(Path(file).read_bytes()).hexdigest())
    return ensemble,digests


def run(checkpoints=(),out="runs/world3d-demo",seed=95000,
        context=5,horizon=5,size=24,untrained_smoke=False,active=True):
    if context<2 or horizon<1:
        raise ValueError("Need context and horizon")
    models,digests=load(checkpoints,untrained_smoke)
    record=world3d.episode(seed,context+horizon,size,"ood_camera")
    observed=jnp.asarray(record["rgb"][:context][None])
    actions=jnp.asarray(record["camera_actions"][None])
    forecast=np.asarray(model.rollout(models[0],observed,actions,horizon,True))[0]
    blind=np.asarray(model.rollout(models[0],observed,actions,horizon,False))[0]
    truth=record["rgb"][context:]
    hold=np.repeat(record["rgb"][context-1:context],horizon,axis=0)
    leads=tuple(range(1,horizon+1))
    score=bench.score([record],forecast[None],context,leads)
    base=bench.score([record],hold[None],context,leads)
    blind_score=bench.score([record],blind[None],context,leads)
    target=Path(out);target.mkdir(parents=True,exist_ok=True)
    (target/"storyboard.html").write_text(world3d_storyboard.html(
        record["rgb"][context-1],forecast,truth,hold,
        "Action-conditioned predictive mirror: calibrated 3D world"))
    np.savez_compressed(target/"evidence.npz",
        observed=record["rgb"][:context],planned_actions=record["camera_actions"],
        predicted=forecast,actual=truth,persistence=hold,action_blind=blind)
    active_metrics=None
    if active and len(models)>=2:
        active_metrics=active_world3d.compare(models,(seed,seed+1),
                                              max(3,context),size)
    result={"protocol":"verified-3d-mirror-storyboard-v1",
            "seed":seed,"domain":"ood_camera","context":context,
            "horizon":horizon,"checkpoint_sha256":digests,
            "model_rgb":score,"persistence_rgb":base,
            "action_blind_rgb":blind_score,
            "active_gaze":active_metrics,"untrained_smoke":untrained_smoke,
            "artifacts":["storyboard.html","evidence.npz"],
            "limitations":"Storyboards display withheld future ground truth "
                "AFTER prediction. Analytic spheres are not a real 3D "
                "environment; apparent perspective prediction does not "
                "prove internal 3D structure. Untrained smoke results "
                "must never be cited as learned ability."}
    (target/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints",nargs="*",default=[])
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--no-active",action="store_true")
    p.add_argument("--out",default="runs/world3d-demo")
    p.add_argument("--seed",type=int,default=95000)
    p.add_argument("--context",type=int,default=5)
    p.add_argument("--horizon",type=int,default=5)
    p.add_argument("--size",type=int,default=24)
    a=p.parse_args()
    result=run(a.checkpoints,a.out,a.seed,a.context,a.horizon,
               a.size,a.untrained_smoke,not a.no_active)
    print(json.dumps({"artifacts":result["artifacts"],
                      "out":a.out,"protocol":result["protocol"]},indent=2))


if __name__=="__main__":
    main()
