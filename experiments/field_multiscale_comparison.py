"""Matched parameter-start comparison of multiscale field architectures."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import field_multiscale as net, field_predictors as field
from train import adamw


def run(out="runs/field-multiscale",kind="diffusion",seeds=(11,23,41),
        steps=30,horizon=5,size=20,channels=8,ticks=2,batch_size=4):
    if kind not in ("diffusion","wave") or not seeds or steps<1 or size%2:
        raise ValueError("Need even field sizes and valid configurations")
    train,target,_=field.dataset(range(33000,33024),kind,3,horizon,size)
    validation,vtrue,_=field.dataset(range(63000,63008),kind,3,horizon,size)
    test,truth,_=field.dataset(range(93000,93010),kind,3,horizon,size)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        initial=net.init(jax.random.PRNGKey(int(seed)),channels)
        for mode in ("fine","extra_ticks","dilated","encoder_decoder","bidirectional"):
            params=initial
            m=jax.tree.map(jnp.zeros_like,params)
            v=jax.tree.map(jnp.zeros_like,params)
            def metric(p):
                return float(np.mean(np.square(
                    np.asarray(net.rollout(p,jnp.asarray(validation[:,-1]),
                                           horizon,mode,ticks))-vtrue)))
            best,best_score,best_step=params,metric(params),0
            @jax.jit
            def update(p,mm,vv,batch,labels,index):
                value,grad=jax.value_and_grad(
                    lambda q:net.loss(q,batch,labels,mode,ticks))(p)
                norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grad)))
                grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grad)
                p,mm,vv=adamw(p,grad,mm,vv,index)
                return p,mm,vv
            rng=np.random.default_rng(int(seed)+1000)
            for index in range(1,steps+1):
                choice=rng.integers(0,len(train),batch_size)
                params,m,v=update(params,m,v,jnp.asarray(train[choice,-1]),
                                  jnp.asarray(target[choice]),index)
                if index==steps or index%max(1,steps//5)==0:
                    value=metric(params)
                    if value<best_score:
                        best,best_score,best_step=params,value,index
            forecasts=np.asarray(net.rollout(
                best,jnp.asarray(test[:,-1]),horizon,mode,ticks))
            ablation={}
            if mode=="bidirectional":
                all_arms=net.intervention(best,jnp.asarray(test[:,-1]),horizon,ticks)
                for label in ("no_topdown","no_bottomup"):
                    modified=np.asarray(all_arms[label])
                    ablation[label]={"mse":float(np.mean((modified-truth)**2)),
                                     "prediction_change_l1":float(
                                         np.mean(np.abs(modified-forecasts)))}
            file=folder/f"{mode}-seed{seed}.npz"
            np.savez_compressed(file,**jax.tree.map(np.asarray,best))
            results.append({"mode":mode,"seed":int(seed),
                "checkpoint_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                "selected_step":best_step,"validation_mse":best_score,
                "test":field.metrics(forecasts,truth,test[:,-1],kind),
                "frozen_direction_ablations":ablation})
    report={"protocol":"bidirectional-fields-v1","kind":kind,
            "seeds":list(map(int,seeds)),"steps":steps,"horizon":horizon,
            "results":results,
            "limits":"Same initialized parameter dict and update count, "
              "but different active params/FLOPs and reparameterizations. "
              "Ablation impacts are causal forecast changes, not proof of "
              "improved prediction unless the held-out error also improves. "
              "No substantive run has been asserted."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=("wave","diffusion"),default="diffusion")
    p.add_argument("--steps",type=int,default=30)
    p.add_argument("--out",default="runs/field-multiscale")
    a=p.parse_args()
    r=run(a.out,a.kind,steps=a.steps)
    print(json.dumps({"arms":len(r["results"]),"kind":r["kind"]}))


if __name__=="__main__":
    main()
