"""Numerical reference versus fitted stencil/CNN/shared cellular core."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import field_predictors as field
from train import adamw


def run(out="runs/physical-field-prediction",kind="diffusion",seeds=(11,23,41),
        steps=40,context=3,horizon=5,size=20,channels=8,batch_size=4):
    if kind not in ("diffusion","wave") or not seeds or steps<1 or batch_size<1:
        raise ValueError("Invalid field experiment")
    train,labels,_=field.dataset(range(31000,31040),kind,context,horizon,size)
    validation,validation_labels,_=field.dataset(
        range(61000,61010),kind,context,horizon,size)
    test,truth,test_records=field.dataset(range(91000,91016),kind,context,horizon,size)
    weights=field.fit_stencil(train[:,-2],train[:,-1])
    report={"numerical":field.metrics(field.numerical(
                test_records,context,horizon),truth,test[:,-1],kind),
            "linear_stencil":field.metrics(field.stencil_rollout(
                weights,test[:,-1],horizon,kind),truth,test[:,-1],kind)}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    learned=[]
    for arm in field.ARMS:
        for seed in seeds:
            params=field.init(jax.random.PRNGKey(int(seed)),arm,channels)
            m=jax.tree.map(jnp.zeros_like,params)
            v=jax.tree.map(jnp.zeros_like,params)
            def validation_mse(p):
                prediction=np.asarray(field.rollout(
                    p,jnp.asarray(validation[:,-1]),horizon,arm,kind))
                return float(np.mean((prediction-validation_labels)**2))
            best=params;best_val=validation_mse(params);best_step=0
            @jax.jit
            def update(p,mm,vv,inputs,targets,index):
                loss=lambda q:jnp.mean((field.rollout(
                    q,inputs,horizon,arm,kind)-targets)**2)
                value,grads=jax.value_and_grad(loss)(p)
                norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grads)))
                grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grads)
                p,mm,vv=adamw(p,grads,mm,vv,index)
                return p,mm,vv,value
            rng=np.random.default_rng(int(seed)+800)
            for i in range(1,steps+1):
                ids=rng.integers(0,len(train),batch_size)
                params,m,v,_=update(params,m,v,jnp.asarray(train[ids,-1]),
                                     jnp.asarray(labels[ids]),i)
                if i==steps or i%max(1,steps//5)==0:
                    metric=validation_mse(params)
                    if metric<best_val:
                        best,best_val,best_step=params,metric,i
            predicted=np.asarray(field.rollout(
                best,jnp.asarray(test[:,-1]),horizon,arm,kind))
            file=folder/f"{arm}-seed{seed}.npz"
            np.savez_compressed(file,**jax.tree.map(np.asarray,best))
            learned.append({"arm":arm,"seed":int(seed),
                            "validation_mse":best_val,"selected_step":best_step,
                            "checkpoint_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                            "test":field.metrics(predicted,truth,test[:,-1],kind)})
    report["learned"]=learned
    output={"protocol":"physical-field-prediction-v1","kind":kind,
            "seeds":list(map(int,seeds)),"context":context,"horizon":horizon,
            "steps":steps,"size":size,"channels":channels,
            "train_seed_start":31000,"validation_seed_start":61000,
            "test_seed_start":91000,"results":report,
            "limits":"Full observed u/v available to every arm. Numerical reference "
                     "knows true coefficients; fitted stencil uses train only. "
                     "Same update count is not matched FLOPs. "
                     "No learned superiority claimed without substantive runs."}
    (folder/"metrics.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=("diffusion","wave"),default="diffusion")
    p.add_argument("--steps",type=int,default=40)
    p.add_argument("--out",default="runs/physical-field-prediction")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    a=p.parse_args()
    result=run(a.out,a.kind,a.seeds,a.steps)
    print(json.dumps({"kind":result["kind"],"learned_arms":len(result["results"]["learned"])}))


if __name__=="__main__":
    main()
