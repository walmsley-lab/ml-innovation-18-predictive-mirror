"""Can shared local cells learn to deliver and retain a signed pulse?"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from src import cellular_signal as model
from train import adamw


def batch(start_seed, count, size=16, low=2, high=4):
    if count < 1 or size < 8 or low < 1 or high < low or high >= size:
        raise ValueError("Invalid signal task")
    sources=np.zeros((count,size,size,1),np.float32)
    targets=np.zeros_like(sources)
    labels=np.zeros((count,),np.float32)
    distances=[]
    for i in range(count):
        rng=np.random.default_rng(start_seed+i)
        ay,ax=rng.integers(0,size,size=2)
        candidates=[(y,x) for y in range(size) for x in range(size)
                    if low<=abs(y-ay)+abs(x-ax)<=high]
        by,bx=candidates[int(rng.integers(len(candidates)))]
        sign=float(rng.choice(np.array([-1.,1.])))
        sources[i,ay,ax,0]=sign
        targets[i,by,bx,0]=1.
        labels[i]=sign
        distances.append(abs(by-ay)+abs(bx-ax))
    return sources,targets,labels,distances


def measure(params, scenes, ticks, allow):
    pulse,mask,sign,_=scenes
    decoded=np.asarray(model.target_signal(params,jnp.asarray(pulse),
                                            jnp.asarray(mask),ticks,allow))
    return {"mse":float(np.mean((decoded-sign)**2)),
            "sign_accuracy":float(np.mean(np.sign(decoded)==sign)),
            "mean_abs_signal":float(np.mean(np.abs(decoded))),
            "episodes":len(sign)}


def run(out="runs/cellular-transmission",seeds=(11,23,41),steps=80,
        size=16,ticks=12,batch_size=8,channels=8):
    if not seeds or steps<1 or size<10 or ticks<6 or batch_size<1:
        raise ValueError("Invalid transmission study configuration")
    train=batch(10000,128,size,2,4)
    validation=batch(20000,32,size,2,4)
    test=batch(30000,32,size,5,7)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        initial=model.init(jax.random.PRNGKey(int(seed)),channels)
        arms={}
        for name in ("local_messages","no_messages"):
            arms[name]={"params":initial,
                        "m":jax.tree.map(jnp.zeros_like,initial),
                        "v":jax.tree.map(jnp.zeros_like,initial),
                        "best":initial,
                        "validation":measure(initial,validation,ticks,name=="local_messages")["mse"],
                        "step":0}
        def make_update(enabled):
            @jax.jit
            def update(p,m,v,pulse,destination,sign,index):
                err,grad=jax.value_and_grad(lambda w:model.loss(
                    w,pulse,destination,sign,ticks,enabled))(p)
                norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
                grad=jax.tree.map(lambda g:g/jnp.maximum(1.,norm),grad)
                p,m,v=adamw(p,grad,m,v,index)
                return p,m,v,err
            return update
        updates={"local_messages":make_update(True),"no_messages":make_update(False)}
        rng=np.random.default_rng(int(seed)+900)
        for index in range(1,steps+1):
            ids=rng.integers(0,len(train[2]),size=batch_size)
            selected=[jnp.asarray(train[k][ids]) for k in range(3)]
            for name,arm in arms.items():
                arm["params"],arm["m"],arm["v"],_=updates[name](
                    arm["params"],arm["m"],arm["v"],*selected,index)
            if index==steps or index%max(1,steps//5)==0:
                for name,arm in arms.items():
                    value=measure(arm["params"],validation,ticks,name=="local_messages")["mse"]
                    if value<arm["validation"]:
                        arm["best"],arm["validation"],arm["step"]=arm["params"],value,index
        entry={"seed":int(seed),"arms":{}}
        for name,arm in arms.items():
            good=name=="local_messages"
            entry["arms"][name]={
                "best_step":arm["step"],"validation_mse":arm["validation"],
                "test":measure(arm["best"],test,ticks,good)}
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        results.append(entry)
    report={
        "experiment":"cellular_signed_pulse_transmission",
        "model": "shared_cellular_rule_3x3", "steps":steps,
        "seeds":list(map(int,seeds)),"size":size,"ticks":ticks,
        "train_distance_range":[2,4],"test_distance_range":[5,7],
        "train_episodes":len(train[2]),"test_episodes":len(test[2]),
        "results":results,
        "limits":"A target query mask is visible to the model and the signed "
                 "label is withheld. This measures conditional transmission, "
                 "not autonomous object binding or learned dynamic routing. "
                 "No RGB predictor is pretrained/transferred in this PR."
    }
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/cellular-transmission")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    p.add_argument("--steps",type=int,default=80)
    p.add_argument("--size",type=int,default=16)
    p.add_argument("--ticks",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=8)
    a=p.parse_args()
    result=run(a.out,a.seeds,a.steps,a.size,a.ticks,a.batch_size)
    print(json.dumps(result["results"],indent=2))


if __name__=="__main__":
    main()
