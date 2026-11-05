"""Sparse approximation to frozen NCA state updates, with intervention tests."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import frozen_state_descriptions as frozen, predictive_rgb_nca as nca
from src import rgb_motion_world as world


def load(checkpoint,channels=8,untrained_smoke=False):
    if checkpoint:
        data=Path(checkpoint).read_bytes()
        with np.load(checkpoint,allow_pickle=False) as p:
            params={key:jnp.asarray(p[key]) for key in p.files}
        required=set(nca.init(jax.random.PRNGKey(0),channels))
        if not required.issubset(params):
            raise ValueError("Checkpoint lacks mandatory NCA parameters")
        return params,hashlib.sha256(data).hexdigest(),False
    if not untrained_smoke:
        raise ValueError("Supply frozen trained checkpoint or --untrained-smoke")
    return nca.init(jax.random.PRNGKey(11),channels),"UNTRAINED_SMOKE",True


def frames(start,count,size):
    return [world.episode(start+i,steps=5,size=size,balls=1)["frames"][-1]
            for i in range(count)]


def run(out="runs/frozen-state-descriptions/metrics.json",checkpoint=None,
        untrained_smoke=False,channels=8,size=16,ticks=12,context=3,
        horizon=5):
    if context+horizon>ticks+1 or ticks<5:
        raise ValueError("Trace too short")
    params,digest,smoke=load(checkpoint,channels,untrained_smoke)
    train=np.stack([frozen.hidden_trace(params,x,ticks)
                     for x in frames(31000,20,size)])
    held=np.stack([frozen.hidden_trace(params,x,ticks)
                    for x in frames(91000,10,size)])
    perturbed=np.stack([frozen.hidden_trace(params,x,ticks,perturb=True)
                         for x in frames(91000,10,size)])
    encoding,models=frozen.fit_descriptors(train)
    report={"protocol":"frozen-nca-sparse-dynamics-v1",
            "checkpoint_sha256":digest,"untrained_smoke":smoke,
            "train_seed_start":31000,"test_seed_start":91000,
            "native_cell_ticks":ticks,
            "ordinary":frozen.evaluate(models,encoding,held,context,horizon),
            "heldout_perturbation":frozen.evaluate(
                models,encoding,perturbed,context,horizon),
            "active_equations":{key:model["active_terms"]
                                for key,model in models.items()},
            "limits":"PCA and coefficients trained on frozen ordinary hidden "
                "trajectories. Test images and local hidden pulses are held-out, "
                "but regression omits the image drive. Causal intervention "
                "changes the underlying network, not proof its fitted "
                "equations are the mechanism. Untrained smoke not research evidence."}
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--out",default="runs/frozen-state-descriptions/metrics.json")
    a=p.parse_args()
    r=run(a.out,a.checkpoint,a.untrained_smoke)
    print(json.dumps({"smoke":r["untrained_smoke"],
                      "methods":list(r["ordinary"])}))


if __name__=="__main__":
    main()
