"""Co-learn sender, receiver and content-based reference pointers on a tiny grid."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import cellular_references as model
from train import adamw


def sample(seed,count=48,size=10,low=2,high=4):
    if count<1 or size<8 or high>=2*size:
        raise ValueError("Invalid grid")
    pulse=np.zeros((count,size,size,1),np.float32)
    target=np.zeros_like(pulse)
    sign=np.zeros(count,np.float32)
    for i in range(count):
        rng=np.random.default_rng(seed+i)
        sy,sx=rng.integers(0,size,size=2)
        candidates=[(y,x) for y in range(size) for x in range(size)
                    if low<=abs(y-sy)+abs(x-sx)<=high]
        if not candidates:
            raise ValueError("No source-target pairs in requested range")
        ty,tx=candidates[int(rng.integers(len(candidates)))]
        sign[i]=float(rng.choice(np.array([-1.,1.])))
        pulse[i,sy,sx,0]=sign[i]
        target[i,ty,tx,0]=1
    return pulse,target,sign


def evaluate(params,examples,ticks,mode):
    p,t,y=examples
    predicted=np.asarray(model.rollout(params,jnp.asarray(p),jnp.asarray(t),ticks,mode))
    return {"mse":float(np.mean((predicted-y)**2)),
            "sign_accuracy":float(np.mean(np.sign(predicted)==y)),
            "episodes":len(y)}


def run(out="runs/cellular-references",seeds=(11,23,41),steps=60,
        size=10,ticks=4,channels=8):
    if not seeds or steps<1 or size<8 or ticks<1:
        raise ValueError("Invalid learned-reference experiment")
    train=sample(11000,72,size,2,4)
    val=sample(21000,24,size,2,4)
    test=sample(31000,24,size,5,7)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        first=model.init(jax.random.PRNGKey(int(seed)),channels)
        zeros=lambda:jax.tree.map(jnp.zeros_like,first)
        arms={mode:{"p":first,"m":zeros(),"v":zeros(),"best":first,
                    "step":0,"val":evaluate(first,val,ticks,mode)["mse"]}
              for mode in model.MODES}
        def make_update(mode):
            @jax.jit
            def update(p,m,v,pulse,target,sign,index):
                value,grads=jax.value_and_grad(
                    lambda w:model.loss(w,pulse,target,sign,ticks,mode))(p)
                norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grads)))
                grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grads)
                p,m,v=adamw(p,grads,m,v,index)
                return p,m,v,value
            return update
        updaters={name:make_update(name) for name in model.MODES}
        rng=np.random.default_rng(int(seed)+500)
        for step in range(1,steps+1):
            indices=rng.integers(0,len(train[2]),size=4)
            examples=[jnp.asarray(x[indices]) for x in train]
            for name,arm in arms.items():
                arm["p"],arm["m"],arm["v"],_=updaters[name](
                    arm["p"],arm["m"],arm["v"],*examples,step)
            if step==steps or step%max(1,steps//5)==0:
                for name,arm in arms.items():
                    score=evaluate(arm["p"],val,ticks,name)["mse"]
                    if score<arm["val"]:
                        arm["best"],arm["val"],arm["step"]=arm["p"],score,step
        entry={"seed":int(seed),"arms":{}}
        for name,arm in arms.items():
            _,routing=model.rollout(arm["best"],jnp.asarray(test[0][:1]),
                                    jnp.asarray(test[1][:1]),ticks,name,True)
            weights=np.asarray(routing)
            entropy=float(np.mean(-np.sum(weights*np.log(weights+1e-9),axis=-1)))
            entry["arms"][name]={"best_step":arm["step"],
                                "validation_mse":arm["val"],
                                "test":evaluate(arm["best"],test,ticks,name),
                                "routing_entropy_nats":entropy}
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        results.append(entry)
    report={
        "experiment":"colearned_content_addressed_cellular_messages",
        "seeds":list(map(int,seeds)),"steps":steps,"ticks":ticks,
        "train_distances":[2,4],"test_distances":[5,7],
        "results":results,
        "limitations":"This is dense soft all-to-all routing O((HW)^2), not "
                      "a scalable hard pointer or discovered object identity. "
                      "Destination query is supplied. Uniform routing ignores "
                      "allocated but inactive query/key weights. Sender, router "
                      "and receiver jointly train only in learned/shuffled arms."
    }
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/cellular-references")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=60)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.steps)["results"],indent=2))


if __name__=="__main__":
    main()
