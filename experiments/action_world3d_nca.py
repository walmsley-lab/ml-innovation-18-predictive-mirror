"""Train action-conditioned and action-blind RGB NCAs on matched 3D worlds."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import action_rgb_nca as model,world3d,world3d_benchmark as bench
from train import adamw


def worlds(seed,count,length,size,domain):
    records=[world3d.episode(seed+i,length,size,domain) for i in range(count)]
    return (np.stack([x["rgb"] for x in records]),
            np.stack([x["camera_actions"] for x in records]),records)


def run(out="runs/action-world3d-nca",seeds=(11,23,41),steps=100,
        context=5,horizon=5,size=24,width=8,batch_size=2):
    if not seeds or steps<1 or context<2 or horizon<1 or width<1:
        raise ValueError("Invalid action-world experiment")
    length=context+horizon
    train=worlds(33000,48,length,size,"train")
    validation=worlds(63000,12,length,size,"train")
    test=worlds(83000,20,length,size,"ood_camera")
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    reports=[]
    for seed in seeds:
        initial=model.init(jax.random.PRNGKey(int(seed)),width)
        zeros=lambda:jax.tree.map(jnp.zeros_like,initial)
        def evaluate(params,inputs,aware):
            video,actions,_=inputs
            forecasts=np.asarray(model.rollout(
                params,jnp.asarray(video[:,:context]),
                jnp.asarray(actions),horizon,aware))
            return float(np.square(forecasts-video[:,context:]).mean())
        arms={aware:{"params":initial,"m":zeros(),"v":zeros(),
                     "best":initial,"step":0,
                     "val":evaluate(initial,validation,aware)}
              for aware in (True,False)}
        def make_update(aware):
            @jax.jit
            def step_fn(p,m,v,video,actions,index):
                value,grad=jax.value_and_grad(
                    lambda q:model.loss(q,video,actions,context,horizon,aware))(p)
                norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
                grad=jax.tree.map(lambda g:g/jnp.maximum(1.,norm),grad)
                p,m,v=adamw(p,grad,m,v,index)
                return p,m,v,value
            return step_fn
        updates={aware:make_update(aware) for aware in (True,False)}
        rng=np.random.default_rng(int(seed)+50)
        for index in range(1,steps+1):
            indices=rng.integers(0,len(train[0]),size=batch_size)
            video=jnp.asarray(train[0][indices])
            actions=jnp.asarray(train[1][indices])
            for aware,arm in arms.items():
                arm["params"],arm["m"],arm["v"],_=updates[aware](
                    arm["params"],arm["m"],arm["v"],video,actions,index)
            if index==steps or index%max(1,steps//5)==0:
                for aware,arm in arms.items():
                    value=evaluate(arm["params"],validation,aware)
                    if value<arm["val"]:
                        arm["best"],arm["step"],arm["val"]=arm["params"],index,value
        row={"seed":int(seed),"arms":{}}
        leads=tuple(sorted({horizon,*(i for i in (1,3,5,10) if i<=horizon)}))
        for aware,arm in arms.items():
            mode="action_aware" if aware else "action_blind"
            forecast=np.asarray(model.rollout(
                arm["best"],jnp.asarray(test[0][:,:context]),
                jnp.asarray(test[1]),horizon,aware))
            row["arms"][mode]={
                "selected_step":arm["step"],"validation_mse":arm["val"],
                "test":bench.score(test[2],forecast,context,leads)}
            np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        reports.append(row)
    result={"protocol":"action-conditioned-rgb-world3d-v1",
            "test_domain":"ood_camera","test_seed_root":83000,
            "context":context,"horizon":horizon,"steps":steps,
            "seeds":list(map(int,seeds)),"runs":reports,
            "limitations":"No latent depth supervision, learned policy or "
                "guarantee of metric 3D state. Planned camera actions "
                "are available in both train/test; true future pixels "
                "and depth/pose are withheld. Blind arm is a weaker "
                "information condition, not compute-identical architecture."}
    (folder/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/action-world3d-nca")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=100)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.steps)["runs"],indent=2))


if __name__=="__main__":
    main()
