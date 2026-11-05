"""Stage NCA forecasting difficulty: +1 -> +3 -> +10 frames.

One independent RGB model, existing non-saturating blend output, no framework.
Each stage has separate held-out validation; final test is not consulted here.
"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_grounded_nca as nca
from src import rgb_motion_distribution as world
from src import experiment_harness as harness


def schedule(horizons=(1,3,10), steps_per_stage=100):
    h=tuple(int(x) for x in horizons)
    if (not h or sorted(set(h))!=list(h) or h[0]<1
            or steps_per_stage<1):
        raise ValueError("Expected unique increasing horizons and positive steps")
    return [{"horizon":x,"steps":steps_per_stage} for x in h]


def run(out="runs/rgb-staged-learning",seed=11,
        horizons=(1,3,10),steps_per_stage=100,context=8,
        ticks=2,channels=8,size=32,batch_size=3):
    stages=schedule(horizons,steps_per_stage)
    if context<2 or ticks<1 or size<16 or channels<1 or batch_size<1:
        raise ValueError("Invalid geometry")
    params=nca.init(jax.random.PRNGKey(seed),channels,output="blend")
    zero=lambda:jax.tree.map(jnp.zeros_like,params)
    m,v=zero(),zero()
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=True)
    history=[]
    elapsed_steps=0
    for index,stage in enumerate(stages):
        h=stage["horizon"]
        valid,_=world.scenes(61000+index*1000,8,context+h,size)
        step_update=harness.clipped_adamw_update(
            lambda p,examples:nca.loss(
                p,examples,context,h,ticks,output="blend",foreground_weight=6))
        before=np.asarray(nca.rollout(params,jnp.asarray(valid[:,:context]),
                                      h,ticks,output="blend"))
        before_mse=float(np.mean((before-valid[:,context:])**2))
        for local in range(1,steps_per_stage+1):
            examples,_=world.scenes(
                31000+int(seed)*100000+index*10000+local*batch_size,
                batch_size,context+h,size)
            elapsed_steps+=1
            params,m,v,cost=step_update(
                params,m,v,(jnp.asarray(examples),),elapsed_steps)
            if not np.isfinite(float(cost)):
                raise FloatingPointError("Nonfinite stage objective")
        pred=np.asarray(nca.rollout(params,jnp.asarray(valid[:,:context]),
                                    h,ticks,output="blend"))
        after_mse=float(np.mean((pred-valid[:,context:])**2))
        path=folder/f"stage-{index+1}-h{h}.npz"
        np.savez_compressed(path,**jax.tree.map(np.asarray,params))
        history.append({"stage":index+1,"horizon":h,"train_steps":steps_per_stage,
                        "validation_before_mse":before_mse,
                        "validation_after_mse":after_mse,
                        "validation_seed_base":61000+index*1000,
                        "clip_rail_fraction":float(np.mean(
                            (pred<=1e-6)|(pred>=1-1e-6))),
                        "checkpoint":path.name})
    result={"experiment":"staged_nca_blend_rgb_learning",
            "status":"training study; not final benchmark",
            "seed":seed,"context":context,"steps_total":elapsed_steps,
            "stages":history,"test_evaluated":False,
            "next":"After staged validation, score once on held-out events and stronger controls"}
    (folder/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-staged-learning")
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--horizons",nargs="+",type=int,default=[1,3,10])
    p.add_argument("--steps-per-stage",type=int,default=100)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seed,a.horizons,a.steps_per_stage),indent=2))


if __name__=="__main__":
    main()
