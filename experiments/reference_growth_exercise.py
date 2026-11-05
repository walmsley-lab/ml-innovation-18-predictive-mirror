"""Train and inspect a small pattern-growing NCA; not published benchmark parity."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from src import reference_growth_nca as nca
from src import experiment_harness as harness


def run(out="runs/reference-growth", seed=11, steps=200, size=24,
        channels=8, ticks=16, damage_fraction=.25):
    if steps < 1 or ticks < 1 or size < 8:
        raise ValueError("Invalid budget")
    params = nca.init(jax.random.PRNGKey(seed), channels)
    initial = nca.seed(size=size,channels=channels)
    goal = nca.target(size)[None]
    def loss(p):
        forecast = nca.rgb(nca.rollout(p,initial,ticks))
        return jnp.mean((forecast-goal)**2)
    m = jax.tree.map(jnp.zeros_like,params)
    v = jax.tree.map(jnp.zeros_like,params)
    update = harness.clipped_adamw_update(lambda p:loss(p))
    start_loss = float(loss(params))
    for step in range(1,steps+1):
        params,m,v,training_loss=update(params,m,v,(),step)
        if not np.isfinite(float(training_loss)):
            raise FloatingPointError("Non-finite training loss")
    grown = nca.rollout(params,initial,ticks)
    re_run = nca.rollout(params,nca.damage(grown,damage_fraction),ticks)
    final = np.asarray(nca.rgb(grown))
    recovered = np.asarray(nca.rgb(re_run))
    result = {
        "experiment":"small_reference_inspired_pattern_growth",
        "reference":"Mordvintsev et al. 2020 Distill; not faithful published setting",
        "seed":seed,"training_steps":steps,"ticks":ticks,
        "initial_mse":start_loss,"final_in_sample_mse":float(loss(params)),
        "recovered_after_damage_mse":float(np.mean((
            recovered-goal)**2)),
        "limits":"Self-generated ring target; no reference-paper metric parity claim."
    }
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"sample.npz",seed=np.asarray(initial),
                        grown=final,recovered=recovered,target=np.asarray(goal))
    np.savez_compressed(folder/"parameters.npz",**jax.tree.map(np.asarray,params))
    (folder/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/reference-growth")
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--ticks",type=int,default=16)
    p.add_argument("--size",type=int,default=24)
    p.add_argument("--seed",type=int,default=11)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seed,a.steps,a.size,ticks=a.ticks),indent=2))


if __name__=="__main__":
    main()
