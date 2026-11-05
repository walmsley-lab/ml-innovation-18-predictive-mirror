"""Roadmap 37: train existing RGB phase NCA against matched plain NCA.

Each arm gets identical RGB clips, initialization and optimizer schedule.
Coupling changes computation, not parameter count. Periodic and irregular
motion remain independent held-out tasks, not claims of cortical oscillations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_phase_nca as phase, predictive_rgb_nca as nca, rgb_benchmark
from train import adamw

MODES=phase.MODES


def episode(seed, total=10, size=20, domain="periodic"):
    if total<4 or size<16 or domain not in ("periodic","chirp","nonperiodic"):
        raise ValueError("Bad motion episode")
    rng=np.random.default_rng(int(seed))
    offset=rng.uniform(-np.pi,np.pi)
    yy,xx=np.mgrid[:size,:size]
    frames=[]
    for t in range(total):
        angle=(.55*t+offset if domain=="periodic" else
               .4*t+.025*t*t+offset if domain=="chirp" else
               .7*t+offset+.11*np.sin(1.7*t+offset))
        x=size/2+(size/4)*np.cos(angle)
        y=size/2+(size/4)*np.sin(angle)
        frame=np.full((size,size,3),.035,np.float32)
        disc=(xx-x)**2+(yy-y)**2<=5.
        frame[disc]=[.95,.25,.12]
        frames.append(frame)
    return np.stack(frames)


def pool(start,count,context,horizon,size,domain):
    return np.stack([episode(start+i,context+horizon,size,domain)
                     for i in range(count)])


def mse(params,frames,mode,context,horizon,ticks):
    predicted=phase.rollout(params,jnp.asarray(frames[:,:context]),
                            horizon,ticks,mode)
    return float(jnp.mean((predicted-jnp.asarray(frames[:,context:]))**2))


def run(out="runs/roadmap-37",seeds=(11,23,41),steps=30,
        context=5,horizon=3,size=20,width=6,ticks=1,batch=2):
    if not seeds or steps<1 or context<2 or horizon<1 or min(size,width,ticks,batch)<1:
        raise ValueError("Invalid matched training")
    train=pool(31000,24,context,horizon,size,"periodic")
    validation=pool(61000,8,context,horizon,size,"periodic")
    heldout={d:pool(91000,12,context,horizon,size,d)
             for d in ("periodic","chirp","nonperiodic")}
    baselines={domain:{
        name:rgb_benchmark.score(prediction,clips,context,(1,horizon))
        for name,prediction in rgb_benchmark.controls(
            clips[:,:context],horizon).items()}
        for domain,clips in heldout.items()}
    output=Path(out) if out else None
    if output: output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        start=nca.init(jax.random.PRNGKey(int(seed)),width)
        indices=np.random.default_rng(int(seed)+22).integers(0,len(train),(steps,batch))
        for mode in MODES:
            params=start
            first=jax.tree.map(jnp.zeros_like,params)
            second=jax.tree.map(jnp.zeros_like,params)
            best=params
            best_step=0
            best_val=mse(params,validation,mode,context,horizon,ticks)
            curve=[{"step":0,"validation_mse":best_val}]
            @jax.jit
            def update(p,m,v,images,step):
                value,grads=jax.value_and_grad(phase.loss)(
                    p,images,context,horizon,ticks,mode)
                norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grads)))
                grads=jax.tree.map(lambda g:g/jnp.maximum(1.,norm),grads)
                p,m,v=adamw(p,grads,m,v,step)
                return p,m,v
            began=time.perf_counter()
            for i,ids in enumerate(indices,1):
                params,first,second=update(params,first,second,
                                           jnp.asarray(train[ids]),i)
                if i==steps or i%max(1,steps//4)==0:
                    v=mse(params,validation,mode,context,horizon,ticks)
                    curve.append({"step":i,"validation_mse":v})
                    if v<best_val:
                        best,best_val,best_step=params,v,i
            jax.block_until_ready(jax.tree.leaves(params)[0])
            runtime=time.perf_counter()-began
            test={}
            for domain,clips in heldout.items():
                prediction=np.asarray(phase.rollout(
                    best,jnp.asarray(clips[:,:context]),horizon,ticks,mode))
                test[domain]=rgb_benchmark.score(prediction,clips,context,(1,horizon))
            checkpoint_hash=None
            if output:
                path=output/f"seed-{seed}-{mode}.npz"
                np.savez_compressed(path,**jax.tree.map(np.asarray,best))
                checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"seed":int(seed),"mode":mode,"best_step":best_step,
                         "validation_mse":best_val,"history":curve,
                         "heldout":test,"training_s_including_jit":runtime,
                         "parameters":int(sum(a.size for a in jax.tree.leaves(best))),
                         "checkpoint_sha256":checkpoint_hash})
    result={"roadmap_id":37,"steps":steps,"context":context,
            "horizon":horizon,"modes":list(MODES),"seeds":list(map(int,seeds)),
            "results":rows,"classical_baselines":baselines,
            "limits":"Matched trainable params/schedules and clips, "
            "but additional oscillator computation and state memory differ. "
            "Short synthetic training does not establish superiority or "
            "neural phase entrainment. Report step-zero selections."}
    if output:(output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-37")
    p.add_argument("--steps",type=int,default=30)
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    a=p.parse_args()
    print(json.dumps({"runs":len(run(a.out,a.seeds,a.steps)["results"])}))


if __name__=="__main__":
    main()
