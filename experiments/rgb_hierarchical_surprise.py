"""Ablate RGB flat recurrence, fast/slow sheets, and latent/surprise losses."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_hierarchical_surprise as model,rgb_benchmark as bench
from train import adamw


def run(out="runs/rgb-hierarchical-surprise",seeds=(11,23,41),steps=60,
        context=8,horizon=10,size=32,batch_size=2,channels=8):
    if not seeds or steps<1 or context<2 or horizon<1 or size%2:
        raise ValueError("Invalid hierarchical RGB study")
    world={"balls":2,"occlusion":True}
    train=bench.frames(bench.scenes(34000,48,context,horizon,size,**world))
    val=bench.frames(bench.scenes(64000,12,context,horizon,size,**world))
    test=bench.frames(bench.scenes(74000,20,context,horizon,size,**world))
    leads=tuple(sorted({horizon,*(h for h in bench.LEADS if h<=horizon)}))
    baselines={k:bench.score(v,test,context,leads)
               for k,v in bench.controls(test[:,:context],horizon).items()}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        initial=model.init(jax.random.PRNGKey(int(seed)),channels)
        zeros=lambda:jax.tree.map(jnp.zeros_like,initial)
        def val_mse(p,mode):
            pred=np.asarray(model.rollout(p,jnp.asarray(val[:,:context]),horizon,mode))
            return float(np.mean((pred-val[:,context:])**2))
        arms={mode:{"p":initial,"m":zeros(),"v":zeros(),"best":initial,
                    "step":0,"val":val_mse(initial,mode)} for mode in model.MODES}
        def update_fn(mode):
            @jax.jit
            def update(p,m,v,examples,index):
                err,grad=jax.value_and_grad(
                    lambda w:model.loss(w,examples,context,horizon,mode))(p)
                norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
                grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grad)
                p,m,v=adamw(p,grad,m,v,index)
                return p,m,v,err
            return update
        updates={mode:update_fn(mode) for mode in model.MODES}
        rng=np.random.default_rng(int(seed)+900)
        for step in range(1,steps+1):
            examples=jnp.asarray(train[rng.integers(0,len(train),size=batch_size)])
            for name,arm in arms.items():
                arm["p"],arm["m"],arm["v"],_=updates[name](
                    arm["p"],arm["m"],arm["v"],examples,step)
            if step==steps or step%max(1,steps//5)==0:
                for name,arm in arms.items():
                    score=val_mse(arm["p"],name)
                    if score<arm["val"]:
                        arm["best"],arm["step"],arm["val"]=arm["p"],step,score
        record={"seed":int(seed),"arms":{}}
        for name,arm in arms.items():
            pred=np.asarray(model.rollout(arm["best"],jnp.asarray(test[:,:context]),
                                          horizon,name))
            record["arms"][name]={"best_step":arm["step"],
                                 "validation_mse":arm["val"],
                                 "test":bench.score(pred,test,context,leads)}
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        results.append(record)
    report={"experiment":"rgb_hierarchical_surprise_latent_ablation",
            "seeds":list(map(int,seeds)),"steps":steps,"context":context,
            "horizon":horizon,"modes":list(model.MODES),"results":results,
            "baselines":baselines,"test_seed_root":74000,
            "limitations":"Learned error gain is not calibrated uncertainty. "
                          "Auxiliary latent teacher is a stop-gradient RGB "
                          "embedding, not a JEPA objective or semantics. "
                          "Different modes have different active compute/capacity. "
                          "Measured held-out gains are required for promotion."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-hierarchical-surprise")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=60)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.steps)["results"],indent=2))


if __name__=="__main__":
    main()
