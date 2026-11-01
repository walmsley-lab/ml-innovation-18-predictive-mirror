"""Paired one-second RGB NCA oscillator ablation on fixed synthetic splits."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as base, rgb_phase_nca as phase
from src import rgb_benchmark as benchmark
from train import adamw


def run(output="runs/rgb-phase-coupling", seeds=(11,23,41), steps=60,
        context=8, horizon=10, ticks=2, size=32, channels=12):
    if not seeds or steps<1 or context<2 or horizon<1:
        raise ValueError("Invalid settings")
    train=benchmark.frames(benchmark.scenes(31000,12,context,horizon,size))
    validation=benchmark.frames(benchmark.scenes(61000,4,context,horizon,size))
    test=benchmark.frames(benchmark.scenes(71000,4,context,horizon,size))
    leads=tuple(sorted({horizon,*(h for h in benchmark.LEADS if h<=horizon)}))
    controls={name:benchmark.score(values,test,context,leads)
              for name,values in benchmark.controls(test[:,:context],horizon).items()}
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        initial=base.init(jax.random.PRNGKey(int(seed)),channels)
        zeros=lambda:jax.tree.map(jnp.zeros_like,initial)
        arms={mode:{"params":initial,"m":zeros(),"v":zeros(),
                    "best":initial,"step":0,"val":float("inf")}
              for mode in phase.MODES}
        def make_update(mode):
            @jax.jit
            def update(p,m,v,x,step):
                error,grads=jax.value_and_grad(
                    lambda q:phase.loss(q,x,context,horizon,ticks,mode))(p)
                norm=jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
                grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grads)
                p,m,v=adamw(p,grads,m,v,step)
                return p,m,v,error
            return update
        updates={mode:make_update(mode) for mode in phase.MODES}
        rng=np.random.default_rng(int(seed)+1000)
        for step in range(1,steps+1):
            examples=jnp.asarray(train[rng.integers(len(train),size=2)])
            for mode,arm in arms.items():
                arm["params"],arm["m"],arm["v"],_=updates[mode](
                    arm["params"],arm["m"],arm["v"],examples,step)
            if step==steps or step%max(1,steps//5)==0:
                for mode,arm in arms.items():
                    prediction=np.asarray(phase.rollout(
                        arm["params"],jnp.asarray(validation[:,:context]),
                        horizon,ticks,mode))
                    metric=float(np.square(prediction-validation[:,context:]).mean())
                    if metric<arm["val"]:
                        arm["best"],arm["val"],arm["step"]=arm["params"],metric,step
        outcomes={}
        for mode,arm in arms.items():
            before=time.perf_counter()
            prediction=np.asarray(phase.rollout(arm["best"],
                               jnp.asarray(test[:,:context]),horizon,ticks,mode))
            duration=(time.perf_counter()-before)*1000
            outcomes[mode]={"best_step":arm["step"],"validation_mse":arm["val"],
                            "test":benchmark.score(prediction,test,context,leads),
                            "batch_eval_wall_ms":duration}
            np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        results.append({"seed":int(seed),"arms":outcomes})
    report={"experiment":"rgb_phase_coupling_ablation","seeds":list(map(int,seeds)),
            "horizon":horizon,"steps":steps,"baselines":controls,"results":results,
            "trainable_parameter_count":int(sum(p.size for p in
                        jax.tree.leaves(base.init(jax.random.PRNGKey(0),channels)))),
            "limitations":"Phase adds fixed state and runtime but zero trainable "
                "parameters. Uncoupled and coupled variants differ by local "
                "Kuramoto-like interactions; neither demonstrates entrainment "
                "benefit without multi-seed held-out improvement. Batch wall "
                "time includes JAX compilation on first call."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-phase-coupling")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=60)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--channels",type=int,default=12)
    a=p.parse_args()
    r=run(a.out,a.seeds,a.steps,a.context,a.horizon,a.ticks,a.size,a.channels)
    print(json.dumps({"experiment":r["experiment"],"seeds":r["seeds"]},indent=2))


if __name__=="__main__":
    main()
