"""Train a small learned CNN baseline under a frozen RGB scene protocol."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import rgb_benchmark as bench, tiny_cnn_video as cnn
from train import adamw


def paired_interval(first, second, truth, lead, seed=93, replicates=500):
    """Episode bootstrap interval for paired difference in per-frame MSE."""
    a=np.mean((first[:,lead-1]-truth[:,lead-1])**2,axis=(1,2,3))
    b=np.mean((second[:,lead-1]-truth[:,lead-1])**2,axis=(1,2,3))
    delta=a-b
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(delta),size=(replicates,len(delta)))
    low,high=np.quantile(delta[draws].mean(axis=1),[.025,.975])
    return {"mean_mse_difference":float(delta.mean()),
            "episode_bootstrap_95pct":[float(low),float(high)],
            "episodes":int(len(delta))}


def run(out="runs/rgb-tiny-cnn",seeds=(11,23,41),steps=60,
        batch_size=2,context=8,horizon=10,size=32,width=16,
        train_count=48,val_count=12,test_count=24,nca_checkpoint=None):
    if not seeds or steps < 1 or batch_size < 1 or context < 2 or horizon < 1:
        raise ValueError("Invalid run configuration")
    family={"balls":2,"occlusion":True}
    total=context+horizon
    train=bench.frames(bench.scenes(31000,train_count,context,horizon,size,**family))
    val=bench.frames(bench.scenes(61000,val_count,context,horizon,size,**family))
    test=bench.frames(bench.scenes(71000,test_count,context,horizon,size,**family))
    if not train_count < 10000 or not val_count < 10000 or not test_count < 10000:
        raise ValueError("Split ranges overlap")
    controls=bench.controls(test[:,:context],horizon)
    lead=horizon
    results=[]
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        initial=cnn.init(jax.random.PRNGKey(int(seed)),width)
        params=initial
        moment=jax.tree.map(jnp.zeros_like,params)
        second=jax.tree.map(jnp.zeros_like,params)
        best=initial
        initial_val=float(cnn.loss(initial,jnp.asarray(val),context,horizon))
        best_val=initial_val
        selected_step=0
        rng=np.random.default_rng(int(seed)+5000)
        @jax.jit
        def update(p,m,v,examples,index):
            value,grad=jax.value_and_grad(
                lambda weights:cnn.loss(weights,examples,context,horizon))(p)
            norm=jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grad)))
            grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grad)
            p,m,v=adamw(p,grad,m,v,index)
            return p,m,v,value
        history=[]
        for step in range(1,steps+1):
            examples=jnp.asarray(train[rng.integers(0,len(train),size=batch_size)])
            params,moment,second,err=update(params,moment,second,examples,step)
            if step==steps or step % max(1,steps//5)==0:
                valid=float(cnn.loss(params,jnp.asarray(val),context,horizon))
                history.append({"step":step,"train_loss":float(err),"validation_mse":valid})
                if valid < best_val:
                    best,best_val,selected_step=params,valid,step
        forecast=np.asarray(cnn.rollout(best,jnp.asarray(test[:,:context]),horizon))
        scores=bench.compare(forecast,test,context,tuple(sorted({lead,*(x for x in bench.LEADS if x<=lead)})))
        interval=paired_interval(forecast,controls["constant_velocity"],
                                 test[:,context:],lead,seed=int(seed))
        np.savez_compressed(folder/f"seed-{seed}-cnn.npz",
                            **jax.tree.map(np.asarray,best))
        results.append({"seed":int(seed),"selected_step":selected_step,
                        "initial_validation_mse":initial_val,
                        "best_validation_mse":best_val,
                        "scores":scores,"vs_velocity":interval,
                        "history":history})
    comparison=None
    if nca_checkpoint:
        from src import predictive_rgb_nca as nca
        with np.load(nca_checkpoint,allow_pickle=False) as file:
            weights={k:jnp.asarray(file[k]) for k in file.files}
        pred=np.asarray(nca.rollout(weights,jnp.asarray(test[:,:context]),horizon))
        comparison={
            "nca_checkpoint":str(nca_checkpoint),
            "sha256":hashlib.sha256(Path(nca_checkpoint).read_bytes()).hexdigest(),
            "scores":bench.score(pred,test,context,(horizon,)),
            "caveat":"Only a fair *test* comparison if the NCA checkpoint was selected "
                     "without accessing these 71000+ episodes and trained on a "
                     "comparable distribution/budget."
        }
    report={
        "experiment":"tiny_cnn_rgb_forecaster",
        "seeds":list(map(int,seeds)),"training_steps":steps,"batch_size":batch_size,
        "train_count":train_count,"validation_count":val_count,"test_count":test_count,
        "train_seed_start":31000,"validation_seed_start":61000,"test_seed_start":71000,
        "world":family,"context":context,"horizon":horizon,
        "results":results,"optional_nca_comparison":comparison,
        "limitations":"This is a tiny CNN residual baseline, not ConvLSTM/SimVP. "
                      "Reported intervals bootstrap independent synthetic episodes "
                      "conditional on this frozen test sample, not model seeds. "
                      "No model superiority is claimed until trained and replicated.",
    }
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-tiny-cnn")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=60)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--width",type=int,default=16)
    p.add_argument("--train-count",type=int,default=48)
    p.add_argument("--val-count",type=int,default=12)
    p.add_argument("--test-count",type=int,default=24)
    p.add_argument("--nca-checkpoint")
    a=p.parse_args()
    data=run(a.out,a.seeds,a.steps,a.batch_size,a.context,a.horizon,a.size,
             a.width,a.train_count,a.val_count,a.test_count,a.nca_checkpoint)
    print(json.dumps({str(x["seed"]):x["scores"]["nca"][str(a.horizon)]
                      for x in data["results"]},indent=2))


if __name__=="__main__":
    main()
