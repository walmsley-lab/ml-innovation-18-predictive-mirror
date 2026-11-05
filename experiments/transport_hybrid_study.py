"""Run transport vs bounded hybrid on matched observed RGB training worlds."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from src import rgb_transport_blend as hybrid
from src import rgb_motion_world as world
from src import experiment_harness as harness


def run(out="runs/transport-hybrid",seed=11,steps=100,
        size=32,context=8,horizon=10,width=12,batch_size=3):
    if min(steps,batch_size,width,horizon)<1 or context<2 or size<16:
        raise ValueError("Invalid training setup")
    validation=world.batch(range(61000,61006),steps=context+horizon,size=size)
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    key=jax.random.PRNGKey(seed)
    initial=hybrid.init(key,width)
    outcomes={}
    for mode in hybrid.MODES:
        params=initial
        m=jax.tree.map(jnp.zeros_like,params)
        v=jax.tree.map(jnp.zeros_like,params)
        update=harness.clipped_adamw_update(
            lambda p,x:hybrid.loss(p,x,context,horizon,mode))
        curve=[]
        for step in range(1,steps+1):
            batch=world.batch(range(11000+seed*100000+step*batch_size,
                                    11000+seed*100000+(step+1)*batch_size),
                              steps=context+horizon,size=size)
            params,m,v,objective=update(params,m,v,(jnp.asarray(batch),),step)
            if not np.isfinite(float(objective)):
                raise FloatingPointError("Nonfinite hybrid objective")
            if step==1 or step==steps or step%max(1,steps//4)==0:
                pred=np.asarray(hybrid.rollout(
                    params,jnp.asarray(validation[:,:context]),horizon,mode))
                curve.append({"step":step,
                    "validation_mse":float(np.mean((pred-validation[:,context:])**2)),
                    "rail_fraction":float(np.mean((pred<=1e-6)|(pred>=1-1e-6)))})
        np.savez_compressed(output/f"{mode}-parameters.npz",
                            **jax.tree.map(np.asarray,params))
        outcomes[mode]={"validation_curve":curve,
                        "checkpoint":f"{mode}-parameters.npz"}
    report={"experiment":"bounded_transport_vs_synthesis",
            "same_seed_and_budget":True,"seed":seed,"steps":steps,
            "status":"validation pilot; no frozen test claims","arms":outcomes}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/transport-hybrid")
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--seed",type=int,default=11)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seed,a.steps),indent=2))


if __name__=="__main__":
    main()
