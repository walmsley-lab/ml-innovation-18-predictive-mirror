"""Roadmap 33 RGB bridge: bounded rehearsal for the existing recurrent NCA.

Each episode first issues a one-frame RGB forecast from observed context.
Only after the next RGB arrives can it become a training sample or enter
the replay reservoir. Both arms spend exactly two gradient steps per episode.
"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_motion_world as world, predictive_rgb_nca as nca

TASKS=("visible_motion","occluded_camera")


def example(seed,task,context=4,size=20):
    if task not in TASKS:
        raise ValueError("Unknown world domain")
    record=world.episode(seed,steps=context+1,size=size,balls=1,
                         occlusion=(task==TASKS[1]),
                         camera_motion=(task==TASKS[1]))
    return record["frames"]


def gradient_step(params,batch,context,ticks,rate=.02):
    if rate <= 0:
        raise ValueError("Positive step size required")
    loss=lambda p:nca.loss(p,jnp.asarray(batch),context,1,ticks)
    _,grad=jax.value_and_grad(loss)(params)
    norm=jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grad)))
    return jax.tree.map(lambda p,g:p-rate*g/jnp.maximum(norm,1.),
                        params,grad)


def evaluate(params,context=4,size=20,ticks=1,per_domain=4):
    out={}
    for j,name in enumerate(TASKS):
        records=np.stack([example(91000+j*100+i,name,context,size)
                          for i in range(per_domain)])
        prediction=np.asarray(nca.rollout(params,jnp.asarray(records[:,:context]),1,ticks))
        out[name]=float(np.square(prediction[:,0]-records[:,context]).mean())
    return out


def learn(seed,mode,capacity=12,stage_lengths=(15,15,8),
          context=4,size=20,channels=4,ticks=1,rate=.02):
    if mode not in ("replay","recent_only") or capacity < 1 or min(stage_lengths)<1:
        raise ValueError("Invalid replay contract")
    initial=nca.init(jax.random.PRNGKey(int(seed)),channels)
    params=initial
    memory=[]
    rng=np.random.default_rng(int(seed)+881)
    seen=0
    replay_count=0
    stage_metrics=[]
    for phase,(task,count) in enumerate(zip((TASKS[0],TASKS[1],TASKS[0]),
                                            stage_lengths)):
        prequential=[]
        for i in range(count):
            current=example(31000+int(seed)*1000+phase*200+i,
                            task,context,size)
            context_only=jnp.asarray(current[:context][None])
            predicted=np.asarray(nca.rollout(params,context_only,1,ticks))[0,0]
            # The target is accessed only after the committed forecast.
            prequential.append(float(np.square(predicted-current[context]).mean()))
            params=gradient_step(params,current[None],context,ticks,rate)
            if mode=="replay" and memory:
                index=int(rng.integers(len(memory)))
                source_index,old=memory[index]
                if source_index >= seen:
                    raise AssertionError("Future replay leakage")
                params=gradient_step(params,old[None],context,ticks,rate)
                replay_count+=1
            else:
                params=gradient_step(params,current[None],context,ticks,rate)
            # Keep one previous-completed episode per reservoir slot.
            if len(memory)<capacity:
                memory.append((seen,current.copy()))
            else:
                slot=int(rng.integers(0,seen+1))
                if slot<capacity:
                    memory[slot]=(seen,current.copy())
            seen+=1
        stage_metrics.append({"stage":phase,"task":task,
                              "examples_seen":seen,
                              "prequential_mse":float(np.mean(prequential)),
                              "heldout_test":evaluate(params,context,size,ticks)})
    return {"seed":int(seed),"mode":mode,"memory_capacity":capacity,
            "total_examples":seen,"gradient_updates":2*seen,
            "replay_updates":replay_count,"stage_metrics":stage_metrics,
            "parameters":params}


def run(out="runs/roadmap-33-rgb",seeds=(11,23,41),
        capacity=12,stage_lengths=(15,15,8),context=4,size=20,
        channels=4,ticks=1):
    folder=Path(out) if out is not None else None
    if folder is not None:
        folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for seed in seeds:
        for mode in ("recent_only","replay"):
            result=learn(int(seed),mode,capacity,stage_lengths,
                         context,size,channels,ticks)
            if folder is not None:
                np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                    **jax.tree.map(np.asarray,result["parameters"]))
            entries.append({k:v for k,v in result.items() if k!="parameters"})
    aggregate={}
    for mode in ("recent_only","replay"):
        arms=[r for r in entries if r["mode"]==mode]
        aggregate[mode]={
            "A_after_B_rgb_mse":float(np.mean([r["stage_metrics"][1]
                         ["heldout_test"][TASKS[0]] for r in arms])),
            "B_after_B_rgb_mse":float(np.mean([r["stage_metrics"][1]
                         ["heldout_test"][TASKS[1]] for r in arms])),
            "A_after_return_rgb_mse":float(np.mean([r["stage_metrics"][2]
                         ["heldout_test"][TASKS[0]] for r in arms]))}
    report={"roadmap_id":33,"protocol":"rgb-nca-rehearsal-v1",
            "seed_list":list(map(int,seeds)),
            "stage_lengths":list(stage_lengths),"context":context,"size":size,
            "channels":channels,"ticks":ticks,"capacity":capacity,
            "updates_per_new_example":2,"aggregate":aggregate,"per_seed":entries,
            "limits":"Actual recurrent RGB NCA under synthetic task A→B→A. "
                     "Online replay is causal and update-matched, but memory "
                     "reads and storage are not cost-matched. Small source/model "
                     "budgets are functional tests, not proof of memory benefit."}
    if folder is not None:
        (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-33-rgb")
    p.add_argument("--capacity",type=int,default=12)
    a=p.parse_args()
    print(json.dumps(run(a.out,capacity=a.capacity)["aggregate"],indent=2))


if __name__=="__main__":
    main()
