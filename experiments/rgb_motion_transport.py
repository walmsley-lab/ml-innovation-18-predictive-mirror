"""Matched RGB residual-only, transport-only and transport+correction study."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_motion_transport as model,rgb_benchmark as bench
from src import experiment_harness as harness


def run(out="runs/rgb-motion-transport",seeds=(11,23,41),steps=80,
        context=8,horizon=10,size=32,batch_size=2,width=12):
    if not seeds or steps<1 or context<2 or horizon<1 or batch_size<1:
        raise ValueError("Invalid transport study")
    family={"balls":2,"occlusion":True}
    train=bench.frames(bench.scenes(33000,64,context,horizon,size,**family))
    val=bench.frames(bench.scenes(63000,12,context,horizon,size,**family))
    test=bench.frames(bench.scenes(73000,24,context,horizon,size,**family))
    leads=tuple(sorted({horizon,*(h for h in bench.LEADS if h<=horizon)}))
    baselines={name:bench.score(values,test,context,leads)
               for name,values in bench.controls(test[:,:context],horizon).items()}
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        initial=model.init(jax.random.PRNGKey(int(seed)),width)
        zero=lambda:jax.tree.map(jnp.zeros_like,initial)
        def validation(p,mode):
            p=np.asarray(model.rollout(p,jnp.asarray(val[:,:context]),horizon,mode))
            return float(np.mean((p-val[:,context:])**2))
        arms={mode:{"p":initial,"m":zero(),"v":zero(),"best":initial,
                    "step":0,"val":validation(initial,mode)} for mode in model.MODES}
        updates={name:harness.clipped_adamw_update(
                     lambda w,examples,mode=name:model.loss(
                         w,examples,context,horizon,mode))
                 for name in model.MODES}
        rng=np.random.default_rng(int(seed)+770)
        for step in range(1,steps+1):
            ids=rng.integers(0,len(train),size=batch_size)
            example=jnp.asarray(train[ids])
            for name,arm in arms.items():
                arm["p"],arm["m"],arm["v"],_=updates[name](
                    arm["p"],arm["m"],arm["v"],(example,),step)
            if step==steps or step%max(1,steps//5)==0:
                for name,arm in arms.items():
                    value=validation(arm["p"],name)
                    if value<arm["val"]:
                        arm["best"],arm["val"],arm["step"]=arm["p"],value,step
        entry={"seed":int(seed),"arms":{}}
        for name,arm in arms.items():
            predicted=np.asarray(model.rollout(
                arm["best"],jnp.asarray(test[:,:context]),horizon,name))
            entry["arms"][name]={
                "selected_step":arm["step"],"validation_mse":arm["val"],
                "test":bench.score(predicted,test,context,leads)}
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        results.append(entry)
    report={"experiment":"rgb_transport_residual_ablation",
            "seeds":list(map(int,seeds)),"steps":steps,"context":context,
            "horizon":horizon,"offsets":model.OFFSETS,
            "test_seed_root":73000,"baselines":baselines,"results":results,
            "limits":"Local shifts are only 0 or +/-2 pixels; no dense optical "
                     "flow or semantic objects. Modes have inactive parameters "
                     "and unequal effective capacity. No claim of advantage "
                     "without held-out scores and timing."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-motion-transport")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=80)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.steps)["results"],indent=2))


if __name__=="__main__":
    main()
