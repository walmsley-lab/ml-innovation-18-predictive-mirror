"""Paired learned architectures on the SAME varied RGB-world repertoire.

No automatic long training. Independent model families, disjoint
train/val/test seeds and fixed evaluation domains; architectures remain
separate. Curriculum (linear-only versus mixed) is the comparison axis.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from src import (rgb_world_ladder as world, rgb_event_metrics as event,
                 predictive_rgb_nca as nca, tiny_cnn_video as cnn,
                 rgb_motion_transport as transport,
                 rgb_soft_references as routing,
                 rgb_hierarchical_surprise as hierarchical)
from train import adamw

MODELS=("cnn","transport","nca","routing","hierarchical")
CURRICULA=("linear","mixed")
TRAIN_DOMAINS=("linear","accelerating","collision","occlusion_short","appearance")
TEST_DOMAINS=("linear","accelerating","collision","occlusion_long",
              "camera_pan","appearance","combined")


def specification(name,key,width,horizon,ticks):
    if name=="cnn":
        return cnn.init(key,width),lambda p,x:cnn.rollout(p,x,horizon)
    if name=="transport":
        return transport.init(key,width),lambda p,x:transport.rollout(
            p,x,horizon,"combined")
    if name=="nca":
        return nca.init(key,width),lambda p,x:nca.rollout(p,x,horizon,ticks)
    if name=="routing":
        return routing.init(key,width),lambda p,x:routing.rollout(
            p,x,horizon,ticks,"learned")
    if name=="hierarchical":
        return hierarchical.init(key,width),lambda p,x:hierarchical.rollout(
            p,x,horizon,"two_scale_rgb")
    if name=="nca_blend":
        try:
            from src import rgb_grounded_nca as grounded
        except ImportError as error:
            raise ValueError("nca_blend needs the independent training-budget-sweep PR") from error
        return (grounded.init(key,width,"blend"),
                lambda p,x:grounded.rollout(p,x,horizon,ticks,"blend"))
    raise ValueError("Unknown model family")


def pool(split,domains,count,context,horizon,size,balls):
    records=[]
    for domain in domains:
        records.extend(world.split(split,count,context,horizon,size,balls,domain))
    return records,np.stack([r["frames"] for r in records])


def objective(p,batch,predictor,context,horizon):
    truth=batch[:,context:context+horizon]
    future=predictor(p,batch[:,:context])
    # Same training objective for every architecture, RGB-only labels.
    fg=jax.nn.sigmoid((jnp.max(truth,axis=-1,keepdims=True)-.12)*40)
    return jnp.mean((1+4*fg)*jnp.square(future-truth))


def report(p,predictor,records,frames,context,horizon):
    leads=tuple(sorted({horizon,*(l for l in (1,3,5,10) if l<=horizon)}))
    forecast=np.asarray(predictor(p,jnp.asarray(frames[:,:context])))
    from src import forecast_diagnostics as diagnostic
    events=event.score(forecast,records,context,leads)
    events["measurement"]=diagnostic.perceptual_failure_report(
        forecast,records,context,leads)
    return events


def run(out="runs/rgb-world-ladder-training",models=MODELS,
        curricula=CURRICULA,seeds=(11,23,41),steps=150,context=8,horizon=10,
        size=32,width=8,ticks=2,batch_size=2,train_per_domain=8,
        val_per_domain=3,test_per_domain=4,train_domains=TRAIN_DOMAINS):
    if (not models or not curricula or not seeds or steps<1 or context<2
            or horizon<1 or size<20 or width<1 or ticks<1 or batch_size<1
            or min(train_per_domain,val_per_domain,test_per_domain)<1):
        raise ValueError("Invalid paired training configuration")
    if (not train_domains or len(set(train_domains)) != len(train_domains)
            or any(domain not in world.DOMAINS for domain in train_domains)):
        raise ValueError("Training domains must be nonempty, distinct and known")
    train_datasets={}
    # Both curricula select checkpoints on identical varied validation
    # scenes; different selection distributions would confound comparison.
    common_validation=pool("validation",TRAIN_DOMAINS,val_per_domain,
                           context,horizon,size,2)
    val_datasets={}
    for curriculum in curricula:
        domains=("linear",) if curriculum=="linear" else tuple(train_domains)
        train_datasets[curriculum]=pool("train",domains,train_per_domain,
                                       context,horizon,size,2)
        val_datasets[curriculum]=common_validation
    test={domain:pool("test",(domain,),test_per_domain,
                      context,horizon,size,2) for domain in TEST_DOMAINS}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in models:
        for seed in seeds:
            rng=np.random.default_rng(int(seed)+7700)
            # Same sampled random indices (modulo pool size) across curricula
            # and architectures; update counts, batch size and objective match.
            sampled=rng.integers(0,2**30,size=(steps,batch_size))
            for curriculum in curricula:
                initial,predictor=specification(
                    name,jax.random.PRNGKey(int(seed)),width,horizon,ticks)
                params=initial
                m=jax.tree.map(jnp.zeros_like,initial)
                v=jax.tree.map(jnp.zeros_like,initial)
                train_records,train_frames=train_datasets[curriculum]
                val_records,val_frames=val_datasets[curriculum]
                def val(p):
                    metrics=report(p,predictor,val_records,val_frames,context,horizon)
                    lead=metrics[str(horizon)]
                    primary=lead["foreground_mse"]
                    return float(primary if primary is not None else lead["mse"]),metrics
                best_val,best_scores=val(initial)
                best_params,best_step=initial,0
                @jax.jit
                def update(p,mm,vv,batch,index):
                    loss=lambda q:objective(q,batch,predictor,context,horizon)
                    value,grad=jax.value_and_grad(loss)(p)
                    norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
                    grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grad)
                    p,mm,vv=adamw(p,grad,mm,vv,index)
                    return p,mm,vv,value
                curve=[{"step":0,"foreground_validation_mse":best_val}]
                start=time.perf_counter()
                for index in range(1,steps+1):
                    ids=sampled[index-1]%len(train_frames)
                    batch=jnp.asarray(train_frames[ids])
                    params,m,v,_=update(params,m,v,batch,index)
                    if index==steps or index%max(1,steps//5)==0:
                        score,metrics=val(params)
                        curve.append({"step":index,"foreground_validation_mse":score})
                        if score<best_val:
                            best_params,best_step,best_val,best_scores=(
                                params,index,score,metrics)
                jax.block_until_ready(jax.tree.leaves(params)[0])
                elapsed=time.perf_counter()-start
                test_reports={domain:report(
                    best_params,predictor,records,frames,context,horizon)
                    for domain,(records,frames) in test.items()}
                stem=f"{name}-seed{seed}-{curriculum}"
                checkpoint=folder/f"{stem}-best.npz"
                np.savez_compressed(checkpoint,
                                    **jax.tree.map(np.asarray,best_params))
                # Last-step training state is distinct from selected checkpoint.
                np.savez_compressed(
                    folder/f"{stem}-train-state.npz",
                    **{"p_"+k:np.asarray(value) for k,value in params.items()},
                    **{"m_"+k:np.asarray(value) for k,value in m.items()},
                    **{"v_"+k:np.asarray(value) for k,value in v.items()})
                results.append({
                    "model":name,"seed":int(seed),"curriculum":curriculum,
                    "selected_step":best_step,"validation_score":best_val,
                    "validation_metric":"foreground_mse at last lead",
                    "validation_leads":best_scores,
                    "history":curve,"test":test_reports,
                    "train_seconds_including_jit":elapsed,
                    "parameter_count":int(sum(x.size for x in jax.tree.leaves(best_params))),
                    "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "last_training_step":steps,"sampler_state":rng.bit_generator.state,
                })
    manifest={"protocol":"paired-diverse-rgb-curriculum-v1",
              "size":size,"width":width,
              "models":list(models),"curricula":list(curricula),
              "seeds":list(map(int,seeds)),"steps":steps,
              "context":context,"horizon":horizon,
              "train_domains":list(train_domains),
              "actual_curriculum_domains":{curriculum: (["linear"] if curriculum=="linear"
                                         else list(train_domains)) for curriculum in curricula},
              "test_domain_seen_in_mixed_training":{domain: domain in train_domains for domain in TEST_DOMAINS},
              "test_domains":list(TEST_DOMAINS),
              "train_seed_root":31000,"validation_seed_root":61000,
              "test_seed_root":91000,"train_per_domain":train_per_domain,
              "validation_per_domain":val_per_domain,
              "test_per_domain":test_per_domain,"results":results,
              "limits":"Every model sees RGB only; camera_pan is an unknown-action "
                  "shift for this arm, not an action-conditioned capability. "
                  "Equal image loss/SGD steps do not imply equal FLOPs or "
                  "capacity. Initializations are model-specific. Matched "
                  "curriculum-exposure counts do not imply identical pixels. "
                  "No significant research advantage is claimed until "
                  "multi-seed comparisons and inference latency are analyzed."}
    (folder/"metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-world-ladder-training")
    p.add_argument("--models",nargs="+",default=list(MODELS))
    p.add_argument("--curricula",nargs="+",choices=CURRICULA,default=list(CURRICULA))
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=150)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--width",type=int,default=8)
    p.add_argument("--train-domains",nargs="+",choices=world.DOMAINS,
                   default=list(TRAIN_DOMAINS))
    a=p.parse_args()
    result=run(a.out,a.models,a.curricula,a.seeds,a.steps,
               a.context,a.horizon,a.size,a.width,
               train_domains=tuple(a.train_domains))
    print(json.dumps({"experiment":result["protocol"],
                      "runs":len(result["results"])},indent=2))


if __name__=="__main__":
    main()
