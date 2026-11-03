"""Frozen causal tests across held-out perspective domains."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import world3d,world3d_causal as causal
from src import world3d_benchmark as bench
from src.active_gaze_comparison import paired_ci


def load(checkpoint=None,untrained_smoke=False):
    if checkpoint is None and not untrained_smoke:
        raise ValueError("Must provide a frozen checkpoint or --untrained-smoke")
    if untrained_smoke:
        return causal.model.init(jax.random.PRNGKey(7),4),"UNTRAINED_SMOKE"
    with np.load(checkpoint,allow_pickle=False) as file:
        params={k:jnp.asarray(file[k]) for k in file.files}
    return params,hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()


def run(checkpoint=None,out="runs/world3d-causal/metrics.json",
        seed=91000,episodes=12,context=5,horizon=5,size=24,
        untrained_smoke=False):
    if episodes<2 or context<2 or horizon<1:
        raise ValueError("Need enough episodes for paired controls")
    params,digest=load(checkpoint,untrained_smoke)
    output={}
    leads=tuple(sorted({horizon,*(h for h in (1,3,5,10) if h<=horizon)}))
    for domain in world3d.DOMAINS:
        records=[world3d.episode(seed+1000*i,context+horizon,size,domain)
                 for i in range(episodes)]
        context_rgb=jnp.asarray(np.stack([r["rgb"][:context] for r in records]))
        actions=jnp.asarray(np.stack([r["camera_actions"] for r in records]))
        outputs={name:np.asarray(causal.rollout(
            params,context_rgb,actions,horizon,name)) for name in causal.MODES}
        results={name:bench.score(records,pred,context,leads)
                 for name,pred in outputs.items()}
        truth=np.stack([r["rgb"][context:] for r in records])
        def per_episode(arr):
            return np.square(arr[:,horizon-1]-truth[:,horizon-1]).mean((1,2,3))
        intact=per_episode(outputs["intact"])
        contrasts={name:paired_ci(per_episode(pred),intact)
                   for name,pred in outputs.items() if name!="intact"}
        output[domain]={"metrics":results,"paired_minus_intact":contrasts}
    report={"protocol":"world3d-causal-ood-v1",
            "checkpoint_sha256":digest,"test_seed_root":seed,
            "episodes_per_domain":episodes,"seeds_increment":1000,
            "domains":output,"context":context,"horizon":horizon,
            "untrained_smoke":bool(untrained_smoke),
            "limits":"Frozen state lesions and action disruptions establish "
                "prediction dependence, not interpretability or necessity "
                "for learning. OOD procedural domains may share textures/"
                "geometry; other simulators and camera recordings remain "
                "future tests. Shifting actions across samples is a null, "
                "not an actionable policy."}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--out",default="runs/world3d-causal/metrics.json")
    p.add_argument("--episodes",type=int,default=12)
    p.add_argument("--seed",type=int,default=91000)
    a=p.parse_args()
    r=run(a.checkpoint,a.out,a.seed,a.episodes,untrained_smoke=a.untrained_smoke)
    print(json.dumps({"protocol":r["protocol"],"domains":list(r["domains"])},indent=2))


if __name__=="__main__":
    main()
