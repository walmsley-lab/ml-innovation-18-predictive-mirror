"""Matched +10-frame ablation of explicit NCA neighborhood selection rules.

No learned reference links; relative and mirror-pointer modes are controls.
"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import selectable_rgb_nca as nca, rgb_benchmark as benchmark
from train import adamw


def run(output="runs/rgb-selector-benchmark", seeds=(11,23,41), steps=60,
        context=8, horizon=10, size=32, ticks=2, channels=12):
    if not seeds or steps<1 or context<2 or horizon<1:
        raise ValueError("Invalid experiment settings")
    train=benchmark.frames(benchmark.scenes(31000,12,context,horizon,size))
    validation=benchmark.frames(benchmark.scenes(61000,4,context,horizon,size))
    test=benchmark.frames(benchmark.scenes(71000,4,context,horizon,size))
    refs=nca.reference_map(size,size)
    leads=tuple(sorted({horizon,*(h for h in benchmark.LEADS if h<=horizon)}))
    controls={name:benchmark.score(value,test,context,leads)
              for name,value in benchmark.controls(test[:,:context],horizon).items()}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for seed in seeds:
        initial=nca.init(jax.random.PRNGKey(int(seed)),channels)
        zeros=lambda:jax.tree.map(jnp.zeros_like,initial)
        arms={mode:{"params":initial,"m":zeros(),"v":zeros(),
                    "best":initial,"best_step":0,"best_val":float("inf")}
              for mode in nca.MODES}

        def make_update(mode):
            ref=refs if mode=="reference" else None
            @jax.jit
            def update(p,m,v,examples,step):
                error,grad=jax.value_and_grad(
                    lambda q:nca.loss(q,examples,context,horizon,ticks,mode,ref))(p)
                norm=jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grad)))
                grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grad)
                p,m,v=adamw(p,grad,m,v,step)
                return p,m,v,error
            return update
        updates={mode:make_update(mode) for mode in nca.MODES}
        rng=np.random.default_rng(int(seed)+1000)
        history=[]
        for step in range(1,steps+1):
            x=jnp.asarray(train[rng.integers(len(train),size=2)])
            for mode in nca.MODES:
                arm=arms[mode]
                arm["params"],arm["m"],arm["v"],_=updates[mode](
                    arm["params"],arm["m"],arm["v"],x,step)
            if step==steps or step%max(1,steps//5)==0:
                row={"step":step}
                for mode in nca.MODES:
                    arm=arms[mode]
                    ref=refs if mode=="reference" else None
                    pred=np.asarray(nca.rollout(arm["params"],
                                               jnp.asarray(validation[:,:context]),
                                               horizon,ticks,mode,ref))
                    metric=float(np.square(pred-validation[:,context:]).mean())
                    row[mode]=metric
                    if metric<arm["best_val"]:
                        arm["best"],arm["best_val"],arm["best_step"]=(
                            arm["params"],metric,step)
                history.append(row)
        outcomes={}
        for mode,arm in arms.items():
            ref=refs if mode=="reference" else None
            forecast=np.asarray(nca.rollout(
                arm["best"],jnp.asarray(test[:,:context]),horizon,ticks,mode,ref))
            outcomes[mode]={"selected_step":arm["best_step"],
                            "validation_mse":arm["best_val"],
                            "test":benchmark.score(forecast,test,context,leads)}
            np.savez_compressed(output/f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        reports.append({"seed":int(seed),"arms":outcomes,"history":history})
    report={"experiment":"rgb_nca_selector_shared_benchmark",
            "seeds":list(map(int,seeds)),"horizon":horizon,"steps":steps,
            "modes":list(nca.MODES),"baselines":controls,"runs":reports,
            "limitations":"References are an externally fixed diagonal mirror, "
              "not learned identities. Local/relative/reference selection differ "
              "in communication density and compute, so causal selector superiority "
              "is not established by MSE alone."}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-selector-benchmark")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    p.add_argument("--steps",type=int,default=60)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--channels",type=int,default=12)
    a=p.parse_args()
    report=run(a.out,a.seeds,a.steps,a.context,a.horizon,a.size,
               a.ticks,a.channels)
    print(json.dumps({"experiment":report["experiment"],
                      "seeds":report["seeds"]},indent=2))


if __name__=="__main__":
    main()
