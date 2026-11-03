"""Compare recurrent NCA, transport-only, and recurrent transport correction."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import recurrent_transport_nca as model, rgb_benchmark as bench
from train import adamw


def run(out="runs/recurrent-transport",seeds=(11,23,41),steps=80,
        context=8,horizon=10,size=32,width=8,batch_size=2):
    if not seeds or steps<1 or context<2 or horizon<1 or width<1:
        raise ValueError("Invalid study")
    world={"balls":2,"occlusion":True}
    train=bench.frames(bench.scenes(35100,48,context,horizon,size,**world))
    validation=bench.frames(bench.scenes(65100,12,context,horizon,size,**world))
    test=bench.frames(bench.scenes(75100,20,context,horizon,size,**world))
    leads=tuple(sorted({horizon,*(h for h in bench.LEADS if h<=horizon)}))
    controls={k:bench.score(v,test,context,leads)
              for k,v in bench.controls(test[:,:context],horizon).items()}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in seeds:
        initial=model.init(jax.random.PRNGKey(int(seed)),width)
        zeros=lambda:jax.tree.map(jnp.zeros_like,initial)
        def score(p,mode,examples):
            prediction=np.asarray(model.rollout(p,jnp.asarray(examples[:,:context]),
                                                horizon,mode))
            return float(np.square(prediction-examples[:,context:]).mean())
        arms={name:{"p":initial,"m":zeros(),"v":zeros(),"best":initial,
                    "best_step":0,"val":score(initial,name,validation)}
              for name in model.MODES}
        def update_for(mode):
            @jax.jit
            def update(p,m,v,examples,index):
                error,grads=jax.value_and_grad(lambda w:model.loss(
                    w,examples,context,horizon,mode))(p)
                norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grads)))
                grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grads)
                p,m,v=adamw(p,grads,m,v,index)
                return p,m,v,error
            return update
        updates={m:update_for(m) for m in model.MODES}
        rng=np.random.default_rng(int(seed)+500)
        for index in range(1,steps+1):
            ids=rng.integers(0,len(train),size=batch_size)
            examples=jnp.asarray(train[ids])
            for mode,arm in arms.items():
                arm["p"],arm["m"],arm["v"],_=updates[mode](
                    arm["p"],arm["m"],arm["v"],examples,index)
            if index==steps or index%max(1,steps//5)==0:
                for mode,arm in arms.items():
                    value=score(arm["p"],mode,validation)
                    if value<arm["val"]:
                        arm["best"],arm["best_step"],arm["val"]=arm["p"],index,value
        record={"seed":int(seed),"modes":{}}
        for mode,arm in arms.items():
            prediction=np.asarray(model.rollout(
                arm["best"],jnp.asarray(test[:,:context]),horizon,mode))
            record["modes"][mode]={
                "selected_step":arm["best_step"],"validation_mse":arm["val"],
                "test":bench.score(prediction,test,context,leads)}
            np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                **{"nca_"+k:np.asarray(v) for k,v in arm["best"]["nca"].items()},
                                **{"transport_"+k:np.asarray(v) for k,v in arm["best"]["transport"].items()})
        runs.append(record)
    report={"protocol":"recurrent-rgb-transport-v1","seeds":list(map(int,seeds)),
            "steps":steps,"context":context,"horizon":horizon,
            "baselines":controls,"results":runs,
            "limitations":"The transport-only control computes inert NCA state and "
            "modes activate different readout paths; equal weight storage "
            "does not mean matched compute. Local offsets only. Train/val/test "
            "are same generator family; event and OOD evaluation needed."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/recurrent-transport")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=80)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.steps)["results"],indent=2))


if __name__=="__main__":
    main()
