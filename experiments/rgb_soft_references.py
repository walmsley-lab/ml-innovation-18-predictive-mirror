"""Matched RGB learned reference-routing vs fixed and broken correspondence."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_soft_references as model, rgb_benchmark as bench
from train import adamw


def run(out="runs/rgb-soft-references",seeds=(11,23,37),steps=40,
        context=8,horizon=10,size=24,ticks=2,channels=8):
    if not seeds or steps<1 or context<2 or horizon<1 or ticks<1:
        raise ValueError("Invalid experiment settings")
    train=bench.frames(bench.scenes(31000,12,context,horizon,size))
    val=bench.frames(bench.scenes(61000,4,context,horizon,size))
    test=bench.frames(bench.scenes(71000,4,context,horizon,size))
    leads=tuple(sorted({horizon,*(h for h in bench.LEADS if h<=horizon)}))
    controls={k:bench.score(v,test,context,leads)
              for k,v in bench.controls(test[:,:context],horizon).items()}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    cases=[]
    for seed in seeds:
        initial=model.init(jax.random.PRNGKey(int(seed)),channels)
        def validation(p,mode):
            prediction=np.asarray(model.rollout(
                p,jnp.asarray(val[:,:context]),horizon,ticks,mode))
            return float(np.square(prediction-val[:,context:]).mean())
        def trainer(mode):
            @jax.jit
            def step(p,m,v,examples,index):
                err,grads=jax.value_and_grad(
                    lambda x:model.loss(x,examples,context,horizon,ticks,mode))(p)
                norm=jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
                grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grads)
                p,m,v=adamw(p,grads,m,v,index)
                return p,m,v,err
            return step
        updates={mode:trainer(mode) for mode in model.MODES}
        arms={}
        for mode in model.MODES:
            start=validation(initial,mode)
            arms[mode]={"params":initial,"m":jax.tree.map(jnp.zeros_like,initial),
                        "v":jax.tree.map(jnp.zeros_like,initial),
                        "best":initial,"step":0,"best_val":start,"step_zero":start}
        rng=np.random.default_rng(int(seed)+1000)
        for step in range(1,steps+1):
            batch=jnp.asarray(train[rng.integers(len(train),size=2)])
            for mode in model.MODES:
                a=arms[mode]
                a["params"],a["m"],a["v"],_=updates[mode](
                    a["params"],a["m"],a["v"],batch,step)
            if step==steps or step%max(1,steps//5)==0:
                for mode,a in arms.items():
                    value=validation(a["params"],mode)
                    if value<a["best_val"]:
                        a["best"],a["step"],a["best_val"]=a["params"],step,value
        row={}
        for mode,a in arms.items():
            output,routing=model.rollout(
                a["best"],jnp.asarray(test[:,:context]),horizon,ticks,mode,True)
            routing=np.asarray(routing)
            entropy=float(np.mean(-np.sum(routing*np.log(routing+1e-9),axis=-1)))
            row[mode]={"selected_step":a["step"],"step_zero_validation_mse":a["step_zero"],
                       "validation_mse":a["best_val"],
                       "mean_routing_entropy_nats":entropy,
                       "test":bench.score(np.asarray(output),test,context,leads)}
            np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray,a["best"]))
        cases.append({"seed":int(seed),"arms":row})
    report={"experiment":"learned_soft_offset_references",
            "training_steps":steps,"seeds":list(map(int,seeds)),
            "horizon":horizon,"context":context,"offsets":model.OFFSETS,
            "baselines":controls,"runs":cases,
            "limitations":"Only nine fixed relative offset candidates; weights "
                          "are learned from current internal state. No object IDs, "
                          "hard pointers, learned arbitrary global connectivity "
                          "or equal runtime guarantee. Uniform router parameters "
                          "are allocated but inactive; step-zero eligible."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-soft-references")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,37])
    p.add_argument("--steps",type=int,default=40)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--size",type=int,default=24)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--channels",type=int,default=8)
    a=p.parse_args()
    result=run(a.out,a.seeds,a.steps,a.context,a.horizon,a.size,a.ticks,a.channels)
    print(json.dumps({"runs":len(result["runs"]),
                      "output":str(Path(a.out)/"metrics.json")},indent=2))


if __name__=="__main__":
    main()
