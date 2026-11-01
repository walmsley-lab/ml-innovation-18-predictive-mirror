"""Paired RGB NCA 1-second objective ablation, matched weights and data."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark
from train import adamw

MODES = ("uniform", "late")


def horizon_weights(horizon, mode):
    if horizon < 2 or mode not in MODES:
        raise ValueError("Need >=2 leads and a known objective")
    weights = np.ones(horizon, np.float32)
    if mode == "late":
        weights[:] = 0.25
        weights[-1] = horizon - (horizon-1)*.25
    return jnp.asarray(weights / weights.mean())


def objective(params, examples, context, horizon, ticks, weights):
    prediction = nca.rollout(params, examples[:, :context], horizon, ticks)
    truth = examples[:, context:context+horizon]
    moving = jnp.mean(jnp.abs(truth-examples[:, context-1:context]),
                      axis=-1,keepdims=True)
    weighted = (1+8*moving)*jnp.square(prediction-truth)*weights[None,:,None,None,None]
    return jnp.mean(weighted)


def run(output="runs/rgb-horizon-objective", seeds=(11,23,41), steps=100,
        context=8, horizon=10, ticks=2, channels=12, size=32):
    if not seeds or steps < 1 or context < 2 or horizon < 2:
        raise ValueError("Invalid experiment")
    # Frozen across arms and initialization seeds; disjoint train/val/test.
    train=benchmark.frames(benchmark.scenes(31000,12,context,horizon,size))
    val=benchmark.frames(benchmark.scenes(61000,5,context,horizon,size))
    test=benchmark.frames(benchmark.scenes(71000,5,context,horizon,size))
    baselines=benchmark.controls(test[:,:context],horizon)
    leads=tuple(h for h in benchmark.LEADS if h<=horizon)

    def updater(mode):
        weights=horizon_weights(horizon,mode)
        @jax.jit
        def update(p,m,v,x,step):
            loss,grads=jax.value_and_grad(
                lambda q:objective(q,x,context,horizon,ticks,weights))(p)
            norm=jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
            grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grads)
            p,m,v=adamw(p,grads,m,v,step)
            return p,m,v,loss
        return update
    updates={mode:updater(mode) for mode in MODES}
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    cases=[]
    for seed in seeds:
        init=nca.init(jax.random.PRNGKey(int(seed)),channels)
        blank=lambda:jax.tree.map(jnp.zeros_like,init)
        arms={name:{"p":init,"m":blank(),"v":blank(),"best":init,
                    "best_step":0,"best_val":float("inf")}
              for name in MODES}
        history=[]
        rng=np.random.default_rng(int(seed)+1000)
        for step in range(1,steps+1):
            chosen=jnp.asarray(train[rng.integers(0,len(train),size=2)])
            for mode in MODES:
                a=arms[mode]
                a["p"],a["m"],a["v"],error=updates[mode](
                    a["p"],a["m"],a["v"],chosen,step)
            if step==steps or step%max(1,steps//5)==0:
                row={"step":step}
                for mode in MODES:
                    a=arms[mode]
                    predicted=np.asarray(nca.rollout(a["p"],jnp.asarray(val[:,:context]),
                                                     horizon,ticks))
                    # Both variants chosen on the SAME plain +10 validation MSE.
                    metric=float(np.square(predicted[:,-1]-val[:,context+horizon-1]).mean())
                    row[mode]=metric
                    if metric<a["best_val"]:
                        a["best"],a["best_val"],a["best_step"]=a["p"],metric,step
                history.append(row)
        outcomes={}
        for mode,a in arms.items():
            pred=np.asarray(nca.rollout(a["best"],jnp.asarray(test[:,:context]),
                                        horizon,ticks))
            outcomes[mode]={"selected_step":a["best_step"],
                            "test":benchmark.score(pred,test,context,leads)}
            np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray,a["best"]))
        cases.append({"seed":int(seed),"arms":outcomes,"history":history})
    baseline={k:benchmark.score(v,test,context,leads) for k,v in baselines.items()}
    report={"experiment":"rgb_one_second_loss_ablation","horizon":horizon,
            "seeds":list(map(int,seeds)),"steps":steps,
            "modes":list(MODES),"results":cases,"baselines":baseline,
            "note":"Matched training examples and parameter budget. "
                   "Late loss emphasizes last horizon without changing NCA. "
                   "Test never used for checkpoint selection."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-horizon-objective")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--channels",type=int,default=12)
    p.add_argument("--size",type=int,default=32)
    a=p.parse_args()
    r=run(a.out,a.seeds,a.steps,a.context,a.horizon,a.ticks,a.channels,a.size)
    print(json.dumps({"experiment":r["experiment"],"seeds":r["seeds"],
                      "baselines":r["baselines"]},indent=2))


if __name__=="__main__":
    main()
