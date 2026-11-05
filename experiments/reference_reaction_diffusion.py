"""Train a tiny local learned rule to forecast Gray-Scott fields."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from src import reference_reaction_diffusion as rd
from src import experiment_harness as harness


def run(out="runs/reference-reaction-diffusion",steps=150,seed=11,
        size=24,frames=12,width=24,horizon=4):
    if steps<1 or frames<3 or horizon<1 or horizon>=frames:
        raise ValueError("Invalid training budget")
    training=rd.sequence(seed,size,frames)
    valid=rd.sequence(seed+11000,size,frames)
    params=rd.init(jax.random.PRNGKey(seed),width)
    zero=lambda:jax.tree.map(jnp.zeros_like,params)
    m,v=zero(),zero()
    x=jnp.asarray(training[:-1][None])
    y=jnp.asarray(training[1:][None])
    update=harness.clipped_adamw_update(
        lambda p,inputs,targets:jnp.mean(
            (rd.predict_one(p,inputs)-targets)**2))
    first=float(jnp.mean((rd.predict_one(params,x)-y)**2))
    for t in range(1,steps+1):
        params,m,v,loss=update(params,m,v,(x,y),t)
        if not np.isfinite(float(loss)):
            raise FloatingPointError("Nonfinite learned PDE loss")
    start=jnp.asarray(valid[:1])
    forecast=np.asarray(rd.rollout(params,start,horizon))[0]
    target=valid[1:horizon+1]
    result={"task":"learned local Gray-Scott fields",
            "reference":"Richardson et al. 2024; not protocol-faithful",
            "initial_training_one_step_mse":first,
            "final_training_one_step_mse":float(jnp.mean(
                (rd.predict_one(params,x)-y)**2)),
            "heldout_rollout_mse":float(np.mean((forecast-target)**2)),
            "seed":seed,"steps":steps,"horizon":horizon,
            "params":"learned shared local transition",
            "limits":"Generated Gray-Scott trajectory is not published PDE benchmark parity"}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"fields.npz",
        observed=np.asarray(start),predicted=forecast,target=target)
    np.savez_compressed(folder/"parameters.npz",**jax.tree.map(np.asarray,params))
    (folder/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/reference-reaction-diffusion")
    p.add_argument("--steps",type=int,default=150)
    p.add_argument("--size",type=int,default=24)
    p.add_argument("--seed",type=int,default=11)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.steps,a.seed,a.size),indent=2))


if __name__=="__main__":
    main()
