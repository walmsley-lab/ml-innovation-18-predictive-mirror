"""Roadmap 32 RGB bridge: prequential adaptation of the real recurrent NCA.

A forecast is computed and scored before its target observation becomes
eligible for an optimizer update. Test sequences are A/B/A synthetic RGB
regimes, not real cameras. CPU smoke budgets are intentionally small.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from src import predictive_rgb_nca as nca, rgb_motion_world
from src.multifuture_motion import render


def stream(seed=11, period=12, context=4, size=32):
    if period<4 or context<2 or size<24:
        raise ValueError("Need visible RGB motion and context")
    rng=np.random.default_rng(seed)
    initial=float(rng.uniform(size*.22,size*.30))
    speed=float(rng.uniform(.45,.65))
    total=context+3*period
    points=[]
    for t in range(total):
        if t < context+period:
            x=initial+speed*t
        elif t < context+2*period:
            x=initial+speed*(context+period)-speed*(t-context-period)
        else:
            x=initial+speed*context+speed*(t-context-2*period)
        points.append([size*.5,x])
    frames=np.stack([render(p,size) for p in points])
    # Regime labels are for scoring only, never included in model inputs.
    return frames,np.repeat(("A","B","A"),period)


def grad_step(params,examples,context,ticks,rate):
    objective=lambda p:nca.loss(p,examples,context,1,ticks)
    _,grads=jax.value_and_grad(objective)(params)
    norm=jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    return jax.tree.map(lambda p,g:p-rate*g/jnp.maximum(norm,1.),params,grads)


def initialize(seed=11,size=24,context=4,ticks=1,pretrain_steps=4,channels=4):
    if pretrain_steps < 0 or channels<1:
        raise ValueError("Invalid pretraining")
    params=nca.init(jax.random.PRNGKey(seed),channels)
    # Independent RGB-only source trajectories, no simulator positions.
    episodes=rgb_motion_world.batch(
        range(31000,31000+max(1,pretrain_steps)),steps=context+1,size=size)
    for step in range(pretrain_steps):
        sample=jnp.asarray(episodes[step:step+1])
        params=grad_step(params,sample,context,ticks,.02)
    return params


def evaluate(params,frames,context=4,ticks=1,rate=.02,mode="online"):
    """One prediction per next frame; current frame consumed only *after* score."""
    if mode not in ("online","frozen") or not 0<rate<=1:
        raise ValueError("Invalid adaptation control")
    frames=np.asarray(frames,np.float32)
    if frames.ndim!=4 or frames.shape[-1]!=3 or len(frames)<=context:
        raise ValueError("Need observed RGB stream")
    model=params
    errors=[]
    change=[]
    for last in range(context-1,len(frames)-1):
        history=jnp.asarray(frames[last-context+1:last+1][None])
        predicted=np.asarray(nca.rollout(model,history,1,ticks))[0,0]
        # Prediction above is committed without ever seeing target RGB.
        future=frames[last+1]
        errors.append(float(np.square(predicted-future).mean()))
        change.append(float(np.mean(np.abs(future-frames[last]))))
        if mode=="online":
            # Only after the score is recorded can target become supervised.
            examples=jnp.asarray(frames[last-context+1:last+2][None])
            model=grad_step(model,examples,context,ticks,rate)
    return {"rgb_mse":np.asarray(errors),
            "changed_fraction":np.asarray(change),
            "last_parameters":model}


def run(out="runs/roadmap-32-rgb",seeds=(11,23,41),
        period=10,context=4,size=24,channels=4,ticks=1,
        pretrain_steps=4,rate=.02):
    if not seeds:
        raise ValueError("Need independent initialization seeds")
    folder=Path(out) if out is not None else None
    if folder is not None:
        folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        initial=initialize(int(seed),size,context,ticks,pretrain_steps,channels)
        frames,_=stream(91000+int(seed),period,context,size)
        arms={}
        for name in ("frozen","online"):
            observed=evaluate(initial,frames,context,ticks,rate,name)
            errors=observed["rgb_mse"]
            if len(errors)!=3*period:
                raise AssertionError("Unexpected prequential alignment")
            arms[name]={
                "A_mse":float(np.mean(errors[:period])),
                "B_mse":float(np.mean(errors[period:2*period])),
                "A_return_mse":float(np.mean(errors[2*period:])),
                "prequential_mean_mse":float(errors.mean()),
                "last_errors":errors.tolist()}
            if folder is not None:
                np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                   **jax.tree.map(np.asarray,observed["last_parameters"]))
        rows.append({"seed":int(seed),"results":arms})
    aggregate={name:{key:float(np.mean([r["results"][name][key]
                                     for r in rows])) for key in
                     ("A_mse","B_mse","A_return_mse","prequential_mean_mse")}
               for name in ("frozen","online")}
    result={"roadmap_id":32,"protocol":"rgb-nca-prequential-adaptation-v1",
            "seeds":list(map(int,seeds)),"period":period,"context":context,
            "size":size,"channels":channels,"ticks":ticks,
            "pretrain_steps":pretrain_steps,"online_step_size":rate,
            "test_seed_rule":"91000+initialization seed",
            "aggregate":aggregate,"per_seed":rows,
            "limits":"Synthetic A/B/A rendered RGB; actual recurrent NCA weights "
                     "update after forecast/score. Frozen control has lower "
                     "optimization cost. No held-out real-camera transfer, "
                     "long-term stability or advantage is claimed."}
    if folder is not None:
        (folder/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-32-rgb")
    p.add_argument("--period",type=int,default=10)
    p.add_argument("--pretrain-steps",type=int,default=4)
    a=p.parse_args()
    print(json.dumps(run(a.out,period=a.period,
                         pretrain_steps=a.pretrain_steps)["aggregate"],indent=2))


if __name__=="__main__":
    main()
