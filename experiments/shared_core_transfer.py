"""Exactly shared cellular convolution across signal pretraining and RGB."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import shared_cellular_core as model, rgb_benchmark as bench
from experiments.cellular_transmission import batch as signals
from train import adamw


def optimizer(loss):
    @jax.jit
    def update(p,m,v,examples,index):
        value,grad=jax.value_and_grad(lambda q:loss(q,*examples))(p)
        norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
        grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grad)
        p,m,v=adamw(p,grad,m,v,index)
        return p,m,v,value
    return update


def fit(start, samples, loss_fn, steps, seed):
    """Same optimizer for both stages. External data/seed explicit."""
    p=start
    m=jax.tree.map(jnp.zeros_like,p)
    v=jax.tree.map(jnp.zeros_like,p)
    update=optimizer(loss_fn)
    rng=np.random.default_rng(seed)
    for index in range(1,steps+1):
        ids=rng.integers(0,len(samples[0]),size=2)
        batch=tuple(jnp.asarray(x[ids]) for x in samples)
        p,m,v,_=update(p,m,v,batch,index)
    return p


def run(out="runs/shared-core-transfer", seeds=(11,23,41), pre_steps=60,
        rgb_steps=60, size=16, context=4, horizon=3, channels=8):
    if (not seeds or min(pre_steps,rgb_steps,channels,horizon) < 1
            or context < 2 or size < 16):
        raise ValueError("Invalid shared-core study")
    train_signal=signals(11100,96,16,2,4)[:3]
    # Distinct seeds for visual examples and signal pulses.
    world={"balls":1,"occlusion":True}
    train_rgb=bench.frames(bench.scenes(31100,24,context,horizon,size,**world))
    validation=bench.frames(bench.scenes(61100,8,context,horizon,size,**world))
    test=bench.frames(bench.scenes(71100,12,context,horizon,size,**world))
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        first=model.init(jax.random.PRNGKey(int(seed)),channels)
        pre=fit(first,train_signal,
                lambda p,x,y,z:model.signal_loss(p,x,y,z,10),
                pre_steps,int(seed)+101)
        # Exactly the same tensor keys and shape are used for both stages.
        shuffled={**pre,"core":jnp.roll(pre["core"],1,axis=0)}
        arms={"scratch":first,"pretrained":pre,"shifted_pretrain":shuffled}
        case={"seed":int(seed),"arms":{}}
        for name,start in arms.items():
            trained=fit(start,(train_rgb,),
                        lambda p,x:model.rgb_loss(p,x,context,horizon),
                        rgb_steps,int(seed)+202)
            predicted=np.asarray(model.rgb(trained,jnp.asarray(test[:,:context]),horizon))
            val=np.asarray(model.rgb(trained,jnp.asarray(validation[:,:context]),horizon))
            case["arms"][name]={
                "validation_mse":float(np.mean((val-validation[:,context:])**2)),
                "test":bench.score(predicted,test,context,(1,horizon)
                                   if horizon>1 else (1,))}
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray,trained))
        results.append(case)
    report={
        "experiment":"shared_core_communication_to_rgb_v1",
        "seeds":list(map(int,seeds)),"pretrain_steps":pre_steps,
        "rgb_steps":rgb_steps,"channels":channels,
        "train_rgb_seed_root":31100,"test_rgb_seed_root":71100,
        "results":results,
        "caveat":"Exact hidden convolution shared in both domains. Scratch has "
                 "fewer total training updates; shifted-pretrain controls "
                 "pretraining exposure but not all effects of weight statistics. "
                 "A compute-matched unrelated-pretraining control and "
                 "validation-selected checkpoints are needed before a strong claim."
    }
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/shared-core-transfer")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--pre-steps",type=int,default=60)
    p.add_argument("--rgb-steps",type=int,default=60)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.pre_steps,a.rgb_steps)["results"],indent=2))


if __name__=="__main__":
    main()
