"""Pretrain × model comparisons with equal total updates and disjoint splits."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from src import structured_pretraining as pre, rgb_world_ladder as world
from src.rgb_benchmark import score
from train import adamw


def target(start,count,context,horizon,size,domain):
    return np.stack([world.episode(start+i,context+horizon,size,balls=2,
                 domain=domain)["frames"] for i in range(count)])


def run(out="runs/structured-pretraining",architectures=("cnn","nca"),
        sources=pre.SOURCES,seeds=(11,23,41),pretrain_steps=30,
        finetune_steps=40,context=5,horizon=3,size=24,width=6,ticks=1,
        batch_size=3):
    if (not seeds or pretrain_steps<1 or finetune_steps<1 or context<2
            or horizon<1 or batch_size<1):
        raise ValueError("Invalid training budget")
    train_target=target(35000,32,context,horizon,size,"occlusion_short")
    validation=target(65000,8,context,horizon,size,"occlusion_short")
    test={d:target(95000,12,context,horizon,size,d)
          for d in ("occlusion_long","accelerating","combined")}
    pretraining={s:pre.samples(33000,32,s,context,horizon,size)
                 for s in sources if s!="scratch"}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in architectures:
        for seed in seeds:
            initialized,predict=pre.architecture(name,jax.random.PRNGKey(
                int(seed)),width,ticks,horizon)
            rng=np.random.default_rng(int(seed)+773)
            all_indices=rng.integers(0,32,(pretrain_steps+finetune_steps,batch_size))
            for source in sources:
                params=initialized
                m=jax.tree.map(jnp.zeros_like,params)
                v=jax.tree.map(jnp.zeros_like,params)
                @jax.jit
                def update(p,mm,vv,batch,index):
                    loss=lambda q:pre.objective(q,batch,context,predict)
                    value,grad=jax.value_and_grad(loss)(p)
                    norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grad)))
                    grad=jax.tree.map(lambda g:g/jnp.maximum(1.,norm),grad)
                    p,mm,vv=adamw(p,grad,mm,vv,index)
                    return p,mm,vv
                began=time.perf_counter()
                # Scratch trains on target for SAME total updates,
                # not only the fine-tune portion.
                for step in range(1,pretrain_steps+finetune_steps+1):
                    if source=="scratch" or step>pretrain_steps:
                        batch=train_target[all_indices[step-1]]
                    else:
                        batch=pretraining[source][all_indices[step-1]]
                    params,m,v=update(params,m,v,jnp.asarray(batch),step)
                    if step==pretrain_steps and source!="scratch":
                        # Compare fine-tuning from a frozen pretraining state.
                        m=jax.tree.map(jnp.zeros_like,m)
                        v=jax.tree.map(jnp.zeros_like,v)
                pred=np.asarray(predict(params,jnp.asarray(validation[:,:context])))
                val=float(np.mean((pred-validation[:,context:])**2))
                by_domain={}
                for domain,images in test.items():
                    forecasts=np.asarray(predict(params,jnp.asarray(images[:,:context])))
                    by_domain[domain]=score(forecasts,images,context,
                                           tuple(range(1,horizon+1)))
                path=folder/f"{name}-{source}-seed{seed}.npz"
                np.savez_compressed(path,**jax.tree.map(np.asarray,params))
                results.append({"architecture":name,"source":source,
                    "seed":int(seed),"val_mse":val,"heldout":by_domain,
                    "total_updates":pretrain_steps+finetune_steps,
                    "source_updates":0 if source=="scratch" else pretrain_steps,
                    "target_updates":pretrain_steps+finetune_steps
                         if source=="scratch" else finetune_steps,
                    "elapsed_s_including_jit":time.perf_counter()-began,
                    "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    report={"protocol":"structured-pretraining-transfer-v1",
            "architectures":list(architectures),"sources":list(sources),
            "seeds":list(map(int,seeds)),"results":results,
            "limits":"Same family/architecture and total updates per arm but "
             "different source pixel statistics and target exposure; "
             "scratch has more target updates. Gradient statistics reset "
             "after pretraining; not matched GPU FLOPs or hours. Synthetic "
             "field stimuli are engineering curricula only, not an animal "
             "development analogy. An architectural advantage requires a "
             "larger preregistered holdout and seed uncertainty."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/structured-pretraining")
    p.add_argument("--architectures",nargs="+",choices=pre.ARCHITECTURES,
                   default=list(pre.ARCHITECTURES))
    p.add_argument("--steps",type=int,default=30)
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    a=p.parse_args()
    r=run(a.out,a.architectures,seeds=a.seeds,pretrain_steps=a.steps)
    print(json.dumps({"runs":len(r["results"])}))


if __name__=="__main__":
    main()
