"""Matched one-second RGB NCA: single fast sheet vs fast+slow feedback."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_multiscale_sheets as model, rgb_benchmark as bench
from train import adamw


def run(out="runs/rgb-multiscale", seeds=(11,23,37), steps=50,
        context=8, horizon=10, size=24, ticks=2, channels=8):
    if not seeds or steps<1 or context<2 or horizon<1 or ticks<1:
        raise ValueError("Invalid seed/training/horizon settings")
    train=bench.frames(bench.scenes(31000,12,context,horizon,size))
    val=bench.frames(bench.scenes(61000,4,context,horizon,size))
    test=bench.frames(bench.scenes(71000,4,context,horizon,size))
    leads=tuple(sorted({horizon,*(h for h in bench.LEADS if h<=horizon)}))
    baselines={k:bench.score(v,test,context,leads)
               for k,v in bench.controls(test[:,:context],horizon).items()}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        initial=model.init(jax.random.PRNGKey(int(seed)),channels)
        def validation(params,mode):
            prediction=np.asarray(model.rollout(
                params,jnp.asarray(val[:,:context]),horizon,ticks,mode))
            return float(np.square(prediction-val[:,context:]).mean())
        def update_fn(mode):
            @jax.jit
            def update(p,m,v,x,i):
                error,grads=jax.value_and_grad(
                    lambda weights:model.loss(weights,x,context,horizon,ticks,mode))(p)
                norm=jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
                grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grads)
                p,m,v=adamw(p,grads,m,v,i)
                return p,m,v,error
            return update
        updates={m:update_fn(m) for m in model.MODES}
        arms={}
        for mode in model.MODES:
            baseline=validation(initial,mode)
            arms[mode]={"params":initial,
                        "m":jax.tree.map(jnp.zeros_like,initial),
                        "v":jax.tree.map(jnp.zeros_like,initial),
                        "best":initial,"step":0,"best_val":baseline,
                        "step_zero_val":baseline}
        rng=np.random.default_rng(int(seed)+1000)
        for step in range(1,steps+1):
            batch=jnp.asarray(train[rng.integers(0,len(train),size=2)])
            for mode,arm in arms.items():
                arm["params"],arm["m"],arm["v"],_=updates[mode](
                    arm["params"],arm["m"],arm["v"],batch,step)
            if step==steps or step%max(1,steps//5)==0:
                for mode,arm in arms.items():
                    metric=validation(arm["params"],mode)
                    if metric<arm["best_val"]:
                        arm["best"],arm["best_val"],arm["step"]=(
                            arm["params"],metric,step)
        reports={}
        for mode,arm in arms.items():
            forecast=np.asarray(model.rollout(
                arm["best"],jnp.asarray(test[:,:context]),horizon,ticks,mode))
            reports[mode]={"selected_step":arm["step"],
                           "step_zero_validation_mse":arm["step_zero_val"],
                           "selected_validation_mse":arm["best_val"],
                           "test":bench.score(forecast,test,context,leads)}
            np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        results.append({"seed":int(seed),"arms":reports})
    summary={"experiment":"fast_vs_fast_slow_rgb_nca",
             "controls":baselines,"runs":results,
             "configuration":{"seeds":list(map(int,seeds)),"steps":steps,
                "context":context,"horizon":horizon,"ticks":ticks,"size":size},
             "limitations":"Both arms allocate identical parameter dictionaries, but "
                           "single arm leaves slow/feedback capacity inactive; "
                           "active parameters, state size and runtime are NOT matched. "
                           "Step-zero checkpoints remain eligible. Test is untouched "
                           "by selection. Fast/slow sheets do not imply brain equivalence."}
    (folder/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-multiscale")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,37])
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--size",type=int,default=24)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--channels",type=int,default=8)
    a=p.parse_args()
    report=run(a.out,a.seeds,a.steps,a.context,a.horizon,a.size,a.ticks,a.channels)
    print(json.dumps({"runs":len(report["runs"]),"out":str(Path(a.out)/"metrics.json")},indent=2))


if __name__=="__main__":
    main()
