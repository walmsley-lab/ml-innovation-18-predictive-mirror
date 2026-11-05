"""Roadmap 41: exact shared-core signal->RGB transfer with strong controls.

Signal and RGB pathways share one hidden convolution tensor exactly. Controls
include signal-label shuffle and scratch trained for the SAME total updates.
Scratch sees more target RGB samples; both update count and target exposure
must be reported, not glossed over as equal-data pretraining.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from src import shared_cellular_core as core, rgb_benchmark as bench
from experiments.cellular_transmission import batch as signal_batch
from experiments.shared_core_transfer import fit


def rgb_pool(root,count,context,horizon,size,scenario):
    return bench.frames(bench.scenes(root,count,context,horizon,size,**scenario))


def run(out="runs/roadmap-41",seeds=(11,23,41),pre_steps=30,
        target_steps=30,context=4,horizon=3,channels=6,size=16):
    if not seeds or pre_steps<1 or target_steps<1 or context<2 or horizon<1 or size!=16:
        raise ValueError("Signal pretraining uses exactly 16x16 fields")
    signal=signal_batch(11100,96,16,2,4)[:3]
    train=rgb_pool(31000,24,context,horizon,size,{"balls":1,"occlusion":True})
    validation=rgb_pool(61000,8,context,horizon,size,{"balls":1,"occlusion":True})
    testing={
        "occlusion":rgb_pool(91000,10,context,horizon,size,{"balls":1,"occlusion":True}),
        "camera_motion":rgb_pool(92000,10,context,horizon,size,{"balls":1,"camera_motion":True}),
        "interactions":rgb_pool(93000,10,context,horizon,size,{"balls":2,"occlusion":True})}
    baselines={domain:{
        name:bench.score(values,clips,context,(1,horizon))
        for name,values in bench.controls(clips[:,:context],horizon).items()}
        for domain,clips in testing.items()}
    folder=Path(out) if out else None
    if folder:folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        initial=core.init(jax.random.PRNGKey(int(seed)),channels)
        randomized=np.random.default_rng(int(seed)+100).permutation(len(signal[2]))
        wrong_sign=(signal[0],signal[1],signal[2][randomized])
        pretrained=fit(initial,signal,
                       lambda p,x,y,z:core.signal_loss(p,x,y,z,10),
                       pre_steps,int(seed)+101)
        mismatched=fit(initial,wrong_sign,
                       lambda p,x,y,z:core.signal_loss(p,x,y,z,10),
                       pre_steps,int(seed)+101)
        starts={"signal_pretrained":pretrained,
                "shuffled_signal_labels":mismatched}
        for name,source in starts.items():
            starts[name]=fit(source,(train,),
                             lambda p,x:core.rgb_loss(p,x,context,horizon),
                             target_steps,int(seed)+202)
        starts["scratch_equal_total_updates"]=fit(
            initial,(train,),lambda p,x:core.rgb_loss(p,x,context,horizon),
            pre_steps+target_steps,int(seed)+202)
        starts["scratch_finetune_only"]=fit(
            initial,(train,),lambda p,x:core.rgb_loss(p,x,context,horizon),
            target_steps,int(seed)+202)
        for name,params in starts.items():
            val_pred=np.asarray(core.rgb(params,jnp.asarray(validation[:,:context]),horizon))
            val_mse=float(np.square(val_pred-validation[:,context:]).mean())
            scores={}
            for domain,clips in testing.items():
                p=np.asarray(core.rgb(params,jnp.asarray(clips[:,:context]),horizon))
                scores[domain]=bench.score(p,clips,context,(1,horizon))
            digest=None
            if folder:
                dest=folder/f"seed-{seed}-{name}.npz"
                np.savez_compressed(dest,**jax.tree.map(np.asarray,params))
                digest=hashlib.sha256(dest.read_bytes()).hexdigest()
            rows.append({"seed":int(seed),"arm":name,"validation_mse":val_mse,
                         "heldout":scores,"checkpoint_sha256":digest,
                         "total_updates":pre_steps+target_steps if
                              name!="scratch_finetune_only" else target_steps,
                         "target_rgb_updates":target_steps if
                              name in ("signal_pretrained","shuffled_signal_labels",
                                       "scratch_finetune_only")
                              else pre_steps+target_steps,
                         "parameters":int(sum(x.size for x in jax.tree.leaves(params)))})
    report={"roadmap_id":41,"seeds":list(map(int,seeds)),
            "pretrain_updates":pre_steps,"finetune_updates":target_steps,
            "results":rows,"classical_baselines":baselines,
            "limits":"Exact hidden cellular core transferred from a "
            "synthetic pulse task, not human neural recordings. Scratch equal-total "
            "sees extra RGB examples, so update parity != exposure parity; "
            "scratch-finetune-only has fewer total updates. Shuffled pulse "
            "labels test transfer specificity. No claimed transfer gain."}
    if folder:(folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-41")
    p.add_argument("--pre-steps",type=int,default=30)
    p.add_argument("--target-steps",type=int,default=30)
    a=p.parse_args()
    print(json.dumps({"rows":len(run(a.out,pre_steps=a.pre_steps,
                                    target_steps=a.target_steps)["results"])}))


if __name__=="__main__":
    main()
