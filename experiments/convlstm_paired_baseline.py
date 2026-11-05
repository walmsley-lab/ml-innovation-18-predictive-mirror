"""Small ConvLSTM baseline trained on RGB paired-history scenes only."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import convlstm_reference as convlstm
from src import paired_occlusion_histories as world
from src import experiment_harness as harness


def samples(seed_base,count,context,horizon):
    return np.stack([r["frames"] for i in range(count)
                     for r in world.pair(seed_base+i,context,horizon)])


def run(out="runs/convlstm-paired",seeds=(11,),steps=100,
        context=8,horizon=10,width=8,train_pairs=16,val_pairs=6):
    if not seeds or steps<1 or min(train_pairs,val_pairs)<2 or width<2:
        raise ValueError("Invalid training budget")
    val=samples(41000,val_pairs,context,horizon)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=[]
    for seed in seeds:
        params=convlstm.init(jax.random.PRNGKey(int(seed)),width)
        zero=lambda:jax.tree.map(jnp.zeros_like,params)
        m,v=zero(),zero()
        update=harness.clipped_adamw_update(
            lambda p,x:convlstm.loss(p,x,context,horizon))
        history=[]
        for t in range(1,steps+1):
            example=samples(11000+t*train_pairs,train_pairs,context,horizon)
            params,m,v,objective=update(params,m,v,(jnp.asarray(example),),t)
            if not np.isfinite(float(objective)):
                raise FloatingPointError("Nonfinite ConvLSTM training loss")
            if t==1 or t==steps or t%max(1,steps//4)==0:
                pred=np.asarray(convlstm.rollout(
                    params,jnp.asarray(val[:,:context]),horizon))
                history.append({"step":t,"validation_mse":
                                float(np.mean((pred-val[:,context:])**2))})
        checkpoint=out/f"seed-{seed}.npz"
        np.savez_compressed(checkpoint,**jax.tree.map(np.asarray,params))
        result.append({"seed":int(seed),"history":history,
                       "checkpoint":checkpoint.name})
    report={"model":"minimal_single_layer_jax_convlstm",
            "reference":"Shi et al. 2015; architecture family only, not full benchmark reproduction",
            "train_pairs_per_step":train_pairs,"validation_pairs":val_pairs,
            "context":context,"horizon":horizon,"steps":steps,
            "status":"trained validation only; frozen test not accessed",
            "runs":result}
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/convlstm-paired")
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--seeds",nargs="+",type=int,default=[11])
    p.add_argument("--width",type=int,default=8)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.steps,width=a.width),indent=2))


if __name__=="__main__":
    main()
