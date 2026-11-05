"""Paired output-head training with factorial oracle diagnostics on held-out video."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from src import layered_output_heads as heads, layered_advection_world as world
from src import classical_transport as classical
from train import adamw


def samples(start,count,context,horizon,size,mode):
    episodes=[world.episode(start+i,context+horizon,size,mode) for i in range(count)]
    return episodes,np.stack([r["frames"] for r in episodes])


def quality(pred,truth,anchor):
    pred,truth=np.asarray(pred),np.asarray(truth)
    mse=np.square(pred-truth).mean(axis=-1)
    changing=np.abs(truth-anchor).mean(axis=-1)>.04
    sharp=float(np.abs(np.diff(pred,axis=2)).mean()+
                np.abs(np.diff(pred,axis=3)).mean())
    return {"rgb_mse":float(mse.mean()),
            "motion_mse":float(mse[changing].mean()) if changing.any() else None,
            "motion_pixels":int(changing.sum()),"sharpness":sharp}


def oracle_probe(params,record,context,mode):
    """Known flow / target visibility used only after model trained and frozen."""
    old,prev,current=[jnp.asarray(record["frames"][i][None])
                      for i in (context-3,context-2,context-1)]
    shape=record["frames"].shape[1:3]
    owner=record["instance_ids"][context]
    flow=np.zeros((*shape,2),np.float32)
    for identity in (0,1):
        flow[owner==identity]=record["velocities"][identity]
    valid=record["target_source_visible"][context-1][None]
    predictions={}
    for use_flow in (False,True):
        for use_valid in (False,True):
            key=f'flow_{int(use_flow)}_visible_{int(use_valid)}'
            out=heads.predict(params,old,prev,current,mode,
                jnp.asarray(flow[None]) if use_flow else None,
                jnp.asarray(valid) if use_valid else None)
            predictions[key]=float(np.mean((np.asarray(out[0])-
                                          record["frames"][context])**2))
    return predictions


def run(out="runs/layered-output-comparison",seeds=(11,23,41),steps=40,
        context=4,horizon=3,size=32,width=8,batch_size=4,
        domain="occlusion"):
    if domain not in world.MODES or not seeds or steps<1 or context<3:
        raise ValueError("Invalid output comparison")
    training,frames=samples(34000,28,context,horizon,size,domain)
    validation,vframes=samples(64000,8,context,horizon,size,domain)
    heldout,testframes=samples(94000,12,context,horizon,size,domain)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        initial=heads.init(jax.random.PRNGKey(int(seed)),width)
        rng=np.random.default_rng(int(seed)+101)
        sample_ids=rng.integers(0,len(frames),(steps,batch_size))
        for mode in heads.MODES:
            p=initial
            m=jax.tree.map(jnp.zeros_like,p);v=jax.tree.map(jnp.zeros_like,p)
            def validate(q):
                pred=np.asarray(heads.rollout(q,jnp.asarray(
                    vframes[:,:context]),horizon,mode))
                return float(np.mean((pred-vframes[:,context:])**2))
            best=p;best_val=validate(p);best_step=0
            @jax.jit
            def update(q,mm,vv,example,index):
                value,grad=jax.value_and_grad(lambda z:heads.loss(
                    z,example,context,horizon,mode))(q)
                norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grad)))
                grad=jax.tree.map(lambda g:g/jnp.maximum(1.,norm),grad)
                q,mm,vv=adamw(q,grad,mm,vv,index)
                return q,mm,vv
            began=time.perf_counter()
            for step in range(1,steps+1):
                p,m,v=update(p,m,v,jnp.asarray(frames[sample_ids[step-1]]),step)
                if step==steps or step%max(1,steps//5)==0:
                    score=validate(p)
                    if score<best_val:
                        best,best_val,best_step=p,score,step
            pred=np.asarray(heads.rollout(best,jnp.asarray(
                testframes[:,:context]),horizon,mode))
            report=quality(pred,testframes[:,context:],
                           testframes[:,context-1:context])
            report["oracle_substitution_2x2"]=oracle_probe(best,heldout[0],context,mode)
            file=folder/f"{mode}-seed{seed}.npz"
            np.savez_compressed(file,**jax.tree.map(np.asarray,best))
            results.append({"seed":int(seed),"head":mode,
                "best_validation_mse":best_val,"selected_step":best_step,
                "test":report,"training_seconds_including_jit":time.perf_counter()-began,
                "parameter_count":int(sum(x.size for x in jax.tree.leaves(best))),
                "checkpoint_sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
    baseline={"persistence":np.stack([classical.forecast(
                rec["frames"][:context],horizon,"persistence") for rec in heldout]),
              "pixel_velocity":np.stack([classical.forecast(
                rec["frames"][:context],horizon,"pixel_velocity") for rec in heldout])}
    final={"protocol":"layered-output-heads-v1","domain":domain,"seeds":list(map(int,seeds)),
           "results":results,"controls":{name:quality(x,testframes[:,context:],
                testframes[:,context-1:context]) for name,x in baseline.items()},
           "limits":"All trained heads see identical past RGB history and share an "
           "initialized core; active head parameter use differs. Oracle flow "
           "and visibility only substitute in held-out diagnostic, never "
           "training/inference. The 2x2 is factorial, NOT additive attribution. "
           "Head synthesis is a local learned residual from past-frame features, "
           "not generative filling from unseen truth."}
    (folder/"metrics.json").write_text(json.dumps(final,indent=2)+"\n")
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/layered-output-comparison")
    parser.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    parser.add_argument("--steps",type=int,default=40)
    a=parser.parse_args()
    print(json.dumps({"runs":len(run(a.out,a.seeds,a.steps)["results"])}))


if __name__=="__main__":
    main()
