"""Train/evaluate an RGB-inferred Gaussian-state NCA without oracle poses."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import world3d,rgb_gaussian_nca as model
from src import world3d_benchmark as benchmark,rgb_benchmark
from train import adamw


def dataset(start,count,context,horizon,size,domain):
    records=[world3d.episode(start+i,context+horizon,size,domain)
             for i in range(count)]
    return (np.stack([r["rgb"] for r in records]),
            np.stack([r["camera_actions"] for r in records]),
            np.asarray(records[0]["K"]),records)


def run(out="runs/rgb-gaussian-nca",seeds=(11,23,41),steps=80,
        context=5,horizon=5,size=24,slots=3,width=8,batch_size=2):
    if (not seeds or steps<1 or context<2 or horizon<1 or
            slots<1 or width<1 or batch_size<1):
        raise ValueError("Invalid RGB Gaussian training budget")
    train=dataset(35000,32,context,horizon,size,"train")
    validation=dataset(65000,8,context,horizon,size,"train")
    test=dataset(95000,12,context,horizon,size,"ood_camera")
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    leads=tuple(sorted({horizon,*(x for x in (1,3,5,10) if x<=horizon)}))
    baseline=rgb_benchmark.controls(test[0][:,:context],horizon)
    runs=[]
    for seed in seeds:
        params=model.init(jax.random.PRNGKey(int(seed)),slots,width)
        m=jax.tree.map(jnp.zeros_like,params)
        v=jax.tree.map(jnp.zeros_like,params)
        def validate(p):
            pred=np.asarray(model.rollout(
                p,jnp.asarray(validation[0][:,:context]),
                jnp.asarray(validation[1]),jnp.asarray(validation[2]),horizon))
            return float(np.mean((pred-validation[0][:,context:])**2))
        best=params
        best_val=validate(best)
        best_step=0
        @jax.jit
        def update(p,mm,vv,rgb,actions,index):
            error,grad=jax.value_and_grad(
                lambda q:model.loss(q,rgb,actions,jnp.asarray(train[2]),
                                    context,horizon))(p)
            norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
            grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grad)
            p,mm,vv=adamw(p,grad,mm,vv,index)
            return p,mm,vv,error
        rng=np.random.default_rng(int(seed)+111)
        history=[{"step":0,"validation_mse":best_val}]
        for index in range(1,steps+1):
            ids=rng.integers(0,len(train[0]),batch_size)
            params,m,v,_=update(params,m,v,jnp.asarray(train[0][ids]),
                                jnp.asarray(train[1][ids]),index)
            if index==steps or index%max(1,steps//5)==0:
                s=validate(params)
                history.append({"step":index,"validation_mse":s})
                if s<best_val:
                    best,best_val,best_step=params,s,index
        row={"seed":int(seed),"selected_step":best_step,
             "validation_curve":history,"modes":{}}
        for mode in model.MODES:
            forecast=np.asarray(model.rollout(
                best,jnp.asarray(test[0][:,:context]),
                jnp.asarray(test[1]),jnp.asarray(test[2]),horizon,mode))
            row["modes"][mode]=benchmark.score(test[3],forecast,context,leads)
        np.savez_compressed(folder/f"seed-{seed}.npz",
                            **{k:np.asarray(value) for k,value in best.items()
                               if k!="dynamics"},
                            **{"dynamics_"+k:np.asarray(value)
                               for k,value in best["dynamics"].items()})
        runs.append(row)
    report={"protocol":"rgb-inferred-gaussian-nca-v1",
            "seeds":list(map(int,seeds)),"steps":steps,
            "context":context,"horizon":horizon,"size":size,
            "slots":slots,"width":width,"train_root":35000,
            "validation_root":65000,"test_root":95000,
            "test_domain":"ood_camera","runs":runs,
            "rgb_baselines":{k:benchmark.score(test[3],v,context,leads)
                            for k,v in baseline.items()},
            "model_inputs":"Only past RGB, fixed public camera calibration "
                "K and commanded camera x/yaw actions. No object centers, "
                "radii, IDs, depth, future images or simulator poses.",
            "limits":"Slot count and camera-origin/action kinematics are "
                "known priors. Color templates and scale inferred from RGB "
                "are not guaranteed to identify individual objects. "
                "Static/velocity Gaussian controls share the trained "
                "encoder; architecture and compute are not matched "
                "against CNN/transport/pixel NCA. No measured wins claimed."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-gaussian-nca")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=80)
    p.add_argument("--slots",type=int,default=3)
    p.add_argument("--width",type=int,default=8)
    a=p.parse_args()
    report=run(a.out,a.seeds,a.steps,slots=a.slots,width=a.width)
    print(json.dumps({"seeds":report["seeds"],"runs":len(report["runs"])},indent=2))


if __name__=="__main__":
    main()
