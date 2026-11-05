"""Focused paired-history object-permanence training and held-out evidence.

No study runs automatically. Seed roots separate training, selection,
and final test. Eval truth is never an inference input.
"""
import argparse
import hashlib
import json
import struct
import time
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import paired_occlusion_histories as world
from src import reappearance_scoring as event
from experiments.rgb_world_ladder_training import specification
from train import adamw

FAMILIES=("cnn","transport","hierarchical","nca","nca_blend","routing")
ROOTS={"train":11000,"validation":41000,"test":71000}


def dataset(split,pairs,context=8,horizon=10,size=32):
    if split not in ROOTS or pairs<1:
        raise ValueError("Invalid paired split")
    records=[r for index in range(pairs)
             for r in world.pair(ROOTS[split]+index,context,horizon,size)]
    return records, np.stack([r["frames"] for r in records])


def _png(path,frame):
    picture=np.asarray(np.clip(frame*255,0,255),np.uint8)
    h,w,_=picture.shape
    raw=b"".join(b"\x00"+picture[y].tobytes() for y in range(h))
    def chunk(k,v):
        return (struct.pack(">I",len(v))+k+v+
                struct.pack(">I",zlib.crc32(k+v)&0xffffffff))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n"+
        chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+
        chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))


def contrast_error(predicted,truth):
    if predicted.shape!=truth.shape or len(predicted)%2:
        raise ValueError("Expected matching ordered left/right episode pairs")
    return float(np.mean(np.square(
        (predicted[::2]-predicted[1::2]) -
        (truth[::2]-truth[1::2]))))


def selection_score(predicted,truth):
    return float(np.mean((predicted-truth)**2)+contrast_error(predicted,truth))


def episode_success(forecast,records,context,location_tolerance=3.):
    """Each return event gets success=1 only for on-time localized mass."""
    output=event.score(forecast,records,context)
    success=[]
    error_penalty=[]
    diagonal=float(np.hypot(*forecast.shape[2:4]))
    for row in output["results"]:
        for item in row["events"]:
            location=item["reappearance_centroid_error_px"]
            timing=item["timing_error_frames"]
            good=(location is not None and location<=location_tolerance
                  and timing is not None and abs(timing)<=1
                  and item["premature_visible_frames"]==0)
            success.append(int(good))
            error_penalty.append(
                float(location) if location is not None else diagonal)
    return {"coverage":len(success),"success_per_event":success,
            "success_rate":float(np.mean(success)),
            "mean_reappearance_location_penalty_px":float(np.mean(error_penalty)),
            "event_metrics":output,
            "criteria":"position error<=3 pixels, return time within one "
                       "frame, no premature appearance, predicted mass present"}


def paired_bootstrap(model_values,baseline_values,seed=7,draws=1000):
    """Resample paired episode differences; CI applies to these held-out pairs."""
    a=np.asarray(model_values,float)
    b=np.asarray(baseline_values,float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2 or draws<2:
        raise ValueError("Need >=2 paired episodes and bootstrap repetitions")
    rng=np.random.default_rng(seed)
    diff=a-b
    samples=diff[rng.integers(0,len(diff),(draws,len(diff)))].mean(axis=1)
    return {"mean_difference":float(diff.mean()),
            "ci_95":[float(x) for x in np.percentile(samples,[2.5,97.5])],
            "pairs":len(diff),"method":"paired episode bootstrap, fixed seed"}


def gallery(folder,name,records,truth,prediction,linear,context):
    """Evidence-only visualization: past, future, prediction, strong control."""
    indices=[0,2,min(4,prediction.shape[1]-1),prediction.shape[1]-1]
    rows=[]
    for i in (0,1):
        observed=records[i]["frames"][context-1]
        cols=[observed,
              *[truth[i,j] for j in indices],
              *[prediction[i,j] for j in indices],
              *[linear[i,j] for j in indices]]
        rows.append(np.concatenate(cols,axis=1))
    image=np.concatenate(rows,axis=0)
    path=Path(folder)/(name+"-gallery.png")
    _png(path,image)
    return path.name


def run(out="runs/object-permanence-campaign",models=FAMILIES,
        seeds=(11,23,41),steps=150,context=8,horizon=10,size=32,
        width=8,ticks=2,batch_pairs=2,train_pairs=24,
        val_pairs=8,test_pairs=12):
    if (not models or not seeds or steps<1 or batch_pairs<1
            or min(train_pairs,val_pairs,test_pairs)<2):
        raise ValueError("Invalid pre-registered campaign budget")
    train,train_rgb=dataset("train",train_pairs,context,horizon,size)
    valid,val_rgb=dataset("validation",val_pairs,context,horizon,size)
    test,test_rgb=dataset("test",test_pairs,context,horizon,size)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    test_truth=test_rgb[:,context:]
    baseline={
        "persistence":np.repeat(test_rgb[:,context-1:context],horizon,axis=1),
        "history_linear":np.stack([world.linear_history_forecast(
            x[:context],horizon) for x in test_rgb])
    }
    controls={name:episode_success(p,test,context)
              for name,p in baseline.items()}
    # Group two opposite directions from the same world seed into one
    # bootstrap unit; the two trajectories are NOT independent episodes.
    def pair_success(scored):
        x=np.asarray(scored["success_per_event"],float)
        if len(x)%2:
            raise ValueError("Expected paired return events")
        return x.reshape((-1,2)).mean(axis=1)
    runs=[]
    for name in models:
        for seed in seeds:
            params,predict=specification(
                name,jax.random.PRNGKey(int(seed)),width,horizon,ticks)
            m=jax.tree.map(jnp.zeros_like,params)
            v=jax.tree.map(jnp.zeros_like,params)
            def forecast(p,observations):
                return np.asarray(predict(p,jnp.asarray(observations[:,:context])))
            def validate(p):
                return selection_score(forecast(p,val_rgb),val_rgb[:,context:])
            best,best_score,selected=params,validate(params),0
            initial_score=best_score
            @jax.jit
            def update(p,mm,vv,full,index):
                target=full[:,context:]
                def loss(q):
                    pred=predict(q,full[:,:context])
                    fg=jax.nn.sigmoid((jnp.max(target,axis=-1,keepdims=True)-.12)*40)
                    rgb=jnp.mean((1+4*fg)*(pred-target)**2)
                    pred_diff=pred[::2]-pred[1::2]
                    truth_diff=target[::2]-target[1::2]
                    return rgb+jnp.mean((pred_diff-truth_diff)**2)
                cost,grad=jax.value_and_grad(loss)(p)
                norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
                grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grad)
                p,mm,vv=adamw(p,grad,mm,vv,index)
                return p,mm,vv,cost
            rng=np.random.default_rng(100000+int(seed))
            start=time.perf_counter()
            history=[{"step":0,"validation_score":initial_score}]
            for step in range(1,steps+1):
                ids=rng.integers(0,train_pairs,size=batch_pairs)
                ordered=np.asarray([v for p in ids for v in (2*p,2*p+1)])
                params,m,v,_=update(params,m,v,jnp.asarray(train_rgb[ordered]),step)
                if step==steps or step%max(1,steps//5)==0:
                    score=validate(params)
                    history.append({"step":step,"validation_score":score})
                    if score<best_score:
                        best,best_score,selected=params,score,step
            for leaf in jax.tree.leaves(params):
                leaf.block_until_ready()
            elapsed=time.perf_counter()-start
            prediction=forecast(best,test_rgb)
            scored=episode_success(prediction,test,context)
            baseline_comparisons={baseline_name:paired_bootstrap(
                pair_success(scored),pair_success(reference),
                seed=int(seed)+101)
                for baseline_name,reference in controls.items()}
            changed=world.controls(test_rgb[:,:context])
            reset_forecast=np.asarray(predict(best,jnp.asarray(changed["reset"])))
            shuffled_forecast=np.asarray(predict(best,jnp.asarray(changed["shuffled"])))
            control_scores={"reset":episode_success(reset_forecast,test,context),
                            "shuffled":episode_success(shuffled_forecast,test,context)}
            history_comparisons={mode:paired_bootstrap(
                pair_success(scored),pair_success(value),seed=int(seed)+102)
                for mode,value in control_scores.items()}
            checkpoint=folder/f"{name}-seed{seed}-best.npz"
            np.savez_compressed(checkpoint,**jax.tree.map(np.asarray,best))
            sample=gallery(folder,f"{name}-seed{seed}",test,
                test_truth,prediction,baseline["history_linear"],context)
            runs.append({
                "model":name,"seed":int(seed),"best_validation_step":selected,
                "validation_score":best_score,"validation_curve":history,
                "train_wall_seconds":elapsed,
                "parameters":int(sum(x.size for x in jax.tree.leaves(best))),
                "checkpoint":checkpoint.name,
                "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "gallery":sample,
                "gallery_sha256":hashlib.sha256((folder/sample).read_bytes()).hexdigest(),
                "test_contrast_mse":contrast_error(prediction,test_truth),
                "test":scored,
                "reset_and_shuffled":control_scores,
                "paired_success_vs_baselines":baseline_comparisons,
                "paired_success_vs_history_controls":history_comparisons})
    manifest={
        "protocol":"focused-history-dependent-permanence-v1",
        "seed_roots":ROOTS,"steps":steps,"seeds":list(map(int,seeds)),
        "models":list(models),"context":context,"horizon":horizon,
        "size":size,"train_pairs":train_pairs,"val_pairs":val_pairs,
        "test_pairs":test_pairs,"batch_pairs":batch_pairs,
        "baselines":{name:{"event":scored,
                          "contrast_mse":contrast_error(baseline[name],test_truth)}
                     for name,scored in controls.items()},
        "runs":runs,
        "limits":"Per-event bootstrap reflects these synthetic held-out "
              "pairs, not a population beyond the simulator. Same RGB-only "
              "train/val/test worlds for all models and controls; training "
              "and validation include paired futures but inference sees "
              "past RGB only. Known-palette linear history control has "
              "explicit world priors. No claim of success before scores exist."}
    (folder/"metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/object-permanence-campaign")
    p.add_argument("--models",nargs="+",choices=FAMILIES,default=list(FAMILIES))
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    p.add_argument("--steps",type=int,default=150)
    p.add_argument("--train-pairs",type=int,default=24)
    p.add_argument("--val-pairs",type=int,default=8)
    p.add_argument("--test-pairs",type=int,default=12)
    a=p.parse_args()
    r=run(a.out,a.models,a.seeds,a.steps,
          train_pairs=a.train_pairs,val_pairs=a.val_pairs,
          test_pairs=a.test_pairs)
    print(json.dumps({"run_count":len(r["runs"]),"test_pair_count":a.test_pairs}))


if __name__=="__main__":
    main()
