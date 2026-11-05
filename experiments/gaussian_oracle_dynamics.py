"""Train local cellular Gaussian dynamics with privileged *observed* geometry."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import world3d,gaussian_splats as splats
from src import gaussian_cellular_dynamics as model
from train import adamw


def worlds(seed,count,context,horizon,size,domain):
    records=[world3d.episode(seed+i,context+horizon,size,domain)
             for i in range(count)]
    before=np.stack([r["centers_world"][context-1] for r in records])
    velocity=np.stack([r["centers_world"][context-1]-
                       r["centers_world"][context-2] for r in records])
    future=np.stack([r["centers_world"][context:] for r in records])
    return before.astype(np.float32),velocity.astype(np.float32),future.astype(np.float32),records


def run(out="runs/gaussian-oracle-dynamics",seeds=(11,23,41),steps=80,
        context=5,horizon=5,size=24,channels=12,batch_size=4):
    if (not seeds or steps<1 or context<2 or horizon<1 or channels<1
            or batch_size<1):
        raise ValueError("Invalid oracle dynamics budget")
    train=worlds(34000,36,context,horizon,size,"train")
    validation=worlds(64000,10,context,horizon,size,"train")
    test=worlds(94000,12,context,horizon,size,"ood_camera")
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    baselines={}
    for name in ("static","velocity"):
        pred=(test[0][:,None] if name=="static" else
              test[0][:,None]+np.arange(1,horizon+1)[None,:,None,None]*test[1][:,None])
        if name=="static":
            pred=np.repeat(pred,horizon,axis=1)
        baselines[name]={"position_rmse_m":float(np.sqrt(np.mean((pred-test[2])**2))),
                         "position_mae_m":float(np.linalg.norm(pred-test[2],axis=-1).mean())}
    results=[]
    for seed in seeds:
        original=model.init(jax.random.PRNGKey(int(seed)),channels)
        params=original
        m=jax.tree.map(jnp.zeros_like,params)
        v=jax.tree.map(jnp.zeros_like,params)
        def val(p):
            pred=np.asarray(model.rollout(p,jnp.asarray(validation[0]),
                                        jnp.asarray(validation[1]),horizon))
            return float(np.mean((pred-validation[2])**2))
        best,selected,best_val=params,0,val(params)
        @jax.jit
        def update(p,mm,vv,centers,velocities,truth,index):
            value,grads=jax.value_and_grad(model.loss)(p,centers,velocities,truth)
            norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grads)))
            grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1),grads)
            p,mm,vv=adamw(p,grads,mm,vv,index)
            return p,mm,vv,value
        rng=np.random.default_rng(int(seed)+2000)
        history=[{"step":0,"validation_mse":best_val}]
        for index in range(1,steps+1):
            ids=rng.integers(0,len(train[0]),batch_size)
            params,m,v,_=update(params,m,v,jnp.asarray(train[0][ids]),
                                jnp.asarray(train[1][ids]),
                                jnp.asarray(train[2][ids]),index)
            if index==steps or index%max(1,steps//5)==0:
                s=val(params)
                history.append({"step":index,"validation_mse":s})
                if s<best_val:
                    best,selected,best_val=params,index,s
        predicted=np.asarray(model.rollout(
            best,jnp.asarray(test[0]),jnp.asarray(test[1]),horizon))
        row={"seed":int(seed),"selected_step":selected,
             "validation_mse":best_val,
             "test_position_rmse_m":float(np.sqrt(np.mean((predicted-test[2])**2))),
             "test_position_mae_m":float(np.linalg.norm(
                 predicted-test[2],axis=-1).mean()),
             "validation_curve":history}
        # Both positions and rendered images are scored for this privileged arm.
        render_error=[]
        for i,rec in enumerate(test[3]):
            state=splats.state_from_record(rec,context)
            frames=splats.render_forecast(rec,predicted[i],context,state)
            render_error.append(float(np.mean(
                (frames-rec["rgb"][context:context+horizon])**2)))
        row["test_rendered_rgb_mse"]=float(np.mean(render_error))
        np.savez_compressed(folder/f"seed-{seed}.npz",
                            **jax.tree.map(np.asarray,best))
        results.append(row)
    report={"protocol":"oracle-gaussian-cellular-dynamics-v1",
            "train_root":34000,"val_root":64000,"test_root":94000,
            "test_domain":"ood_camera","context":context,"horizon":horizon,
            "size":size,"steps":steps,"seeds":list(map(int,seeds)),
            "controls":baselines,"results":results,
            "limits":"Positions/velocities, scales and colors supplied by "
                "simulator during inference: privileged input, not inferred "
                "from RGB. Static/velocity are privileged and must be "
                "compared within this task, not directly to an RGB-only model. "
                "True future centers used only for supervised loss and evaluation. "
                "No paid GPU run or evidence of advantage is claimed."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/gaussian-oracle-dynamics")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=80)
    p.add_argument("--channels",type=int,default=12)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.steps,channels=a.channels)["results"],indent=2))


if __name__=="__main__":
    main()
