"""Does cellular signal pretraining improve RGB forecasting at fixed fine-tune budget?"""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as rgb, rgb_benchmark as bench
from src.transmission_to_rgb import transfer
from train import adamw


def run(communication_checkpoint,out="runs/transmission-rgb-transfer",
        seeds=(11,23,41),steps=60,context=8,horizon=10,size=32,
        batch_size=2,ticks=2):
    if not seeds or steps<1 or context<2 or horizon<1 or batch_size<1:
        raise ValueError("Invalid transfer study")
    with np.load(communication_checkpoint,allow_pickle=False) as archive:
        communication={k:jnp.asarray(archive[k]) for k in archive.files}
    channels=int(communication["readout"].shape[0])
    family={"balls":2,"occlusion":True}
    train=bench.frames(bench.scenes(32000,64,context,horizon,size,**family))
    val=bench.frames(bench.scenes(62000,12,context,horizon,size,**family))
    test=bench.frames(bench.scenes(72000,16,context,horizon,size,**family))
    controls=bench.controls(test[:,:context],horizon)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    @jax.jit
    def update(params,m,v,frames,index):
        error,grads=jax.value_and_grad(lambda p:rgb.loss(
            p,frames,context,horizon,ticks))(params)
        norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grads)))
        grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grads)
        params,m,v=adamw(params,grads,m,v,index)
        return params,m,v,error
    def validate(p):
        forecast=np.asarray(rgb.rollout(p,jnp.asarray(val[:,:context]),horizon,ticks))
        return float(np.mean((forecast-val[:,context:])**2))
    for seed in seeds:
        initial=rgb.init(jax.random.PRNGKey(int(seed)),channels)
        modes={"scratch":initial,
               "transmission":transfer(initial,communication),
               "spatially_shifted_transfer":transfer(initial,communication,shift=1)}
        zeros=lambda:jax.tree.map(jnp.zeros_like,initial)
        arms={name:{"p":start,"m":zeros(),"v":zeros(),"best":start,
                    "step":0,"val":validate(start)}
              for name,start in modes.items()}
        rng=np.random.default_rng(int(seed)+2000)
        for step in range(1,steps+1):
            frames=jnp.asarray(train[rng.integers(0,len(train),size=batch_size)])
            for name,arm in arms.items():
                arm["p"],arm["m"],arm["v"],_=update(
                    arm["p"],arm["m"],arm["v"],frames,step)
            if step==steps or step%max(1,steps//5)==0:
                for name,arm in arms.items():
                    score=validate(arm["p"])
                    if score<arm["val"]:
                        arm["best"],arm["val"],arm["step"]=arm["p"],score,step
        record={"seed":int(seed),"arms":{}}
        leads=tuple(sorted({horizon,*(h for h in bench.LEADS if h<=horizon)}))
        for name,arm in arms.items():
            forecast=np.asarray(rgb.rollout(
                arm["best"],jnp.asarray(test[:,:context]),horizon,ticks))
            record["arms"][name]={
                "best_step":arm["step"],"validation_mse":arm["val"],
                "test":bench.score(forecast,test,context,leads)}
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        results.append(record)
    report={
        "experiment":"cellular_transmission_to_rgb_transfer",
        "communication_checkpoint":str(communication_checkpoint),
        "communication_sha256":hashlib.sha256(Path(communication_checkpoint).read_bytes()).hexdigest(),
        "communication_channels":channels,"seeds":list(map(int,seeds)),
        "fine_tune_steps":steps,"train_episode_root":32000,
        "validation_root":62000,"test_root":72000,"context":context,
        "horizon":horizon,"baselines":{
            key:bench.score(p,test,context,(horizon,)) for key,p in controls.items()},
        "results":results,
        "limits":"Transfer is a **partial heuristic weight mapping** from a different "
                 "architecture, not an exact shared core. The supplied "
                 "communication checkpoint must have been trained on "
                 "separate signal tasks with no RGB-test selection. "
                 "Only measured RGB test gains could support a transfer claim."
    }
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--communication-checkpoint",required=True)
    p.add_argument("--out",default="runs/transmission-rgb-transfer")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=60)
    a=p.parse_args()
    data=run(a.communication_checkpoint,a.out,a.seeds,a.steps)
    print(json.dumps(data["results"],indent=2))


if __name__=="__main__":
    main()
