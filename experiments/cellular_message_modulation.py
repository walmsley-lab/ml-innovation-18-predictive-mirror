"""Two competing pulses: do receiver-side gates help select the relevant one?"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import cellular_message_modulation as model
from train import adamw


def batch(seed,count=64,size=16):
    if count<1 or size<9:
        raise ValueError("Invalid task")
    pulses=np.zeros((count,size,size,1),np.float32)
    cues=np.zeros_like(pulses)
    dest=np.zeros_like(pulses)
    truth=np.zeros((count,),np.float32)
    for index in range(count):
        rng=np.random.default_rng(seed+index)
        target=rng.integers(0,size,size=2)
        allowed=[(y,x) for y in range(size) for x in range(size)
                 if 2<=abs(y-target[0])+abs(x-target[1])<=4]
        relevant=allowed.pop(int(rng.integers(len(allowed))))
        distractor=allowed[int(rng.integers(len(allowed)))]
        sign=float(rng.choice(np.array([-1.,1.])))
        other=float(rng.choice(np.array([-1.,1.])))
        pulses[index,*relevant,0]=sign
        pulses[index,*distractor,0]=other
        cues[index,*relevant,0]=1
        dest[index,*target,0]=1
        truth[index]=sign
    return pulses,cues,dest,truth


def scores(params,examples,ticks,mode):
    signal=np.asarray(model.rollout(params,*[jnp.asarray(x) for x in examples[:3]],
                                    ticks=ticks,mode=mode))[:,-1]
    target=examples[3]
    return {"mse":float(np.mean((signal-target)**2)),
            "relevant_sign_accuracy":float(np.mean(np.sign(signal)==target)),
            "episodes":len(target)}


def run(out="runs/cellular-modulation",seeds=(11,23,41),steps=60,
        size=16,ticks=12,batch_size=8,channels=8):
    if not seeds or steps<1 or size<9 or ticks<1 or batch_size<1:
        raise ValueError("Invalid experiment settings")
    train=batch(41000,96,size)
    validation=batch(51000,24,size)
    test=batch(61000,32,size)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        initial=model.init(jax.random.PRNGKey(int(seed)),channels)
        zeros=lambda:jax.tree.map(jnp.zeros_like,initial)
        arms={mode:{"params":initial,"m":zeros(),"v":zeros(),
                    "best":initial,"step":0,
                    "val":scores(initial,validation,ticks,mode)["mse"]}
              for mode in model.MODES}
        def updater(mode):
            @jax.jit
            def update(p,m,v,pulse,cue,dest,sign,index):
                err,grad=jax.value_and_grad(lambda w:model.loss(
                    w,pulse,cue,dest,sign,ticks,mode))(p)
                norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grad)))
                grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grad)
                p,m,v=adamw(p,grad,m,v,index)
                return p,m,v,err
            return update
        updates={name:updater(name) for name in model.MODES}
        rng=np.random.default_rng(int(seed)+404)
        for step in range(1,steps+1):
            ids=rng.integers(0,len(train[3]),size=batch_size)
            samples=[jnp.asarray(x[ids]) for x in train]
            for name,arm in arms.items():
                arm["params"],arm["m"],arm["v"],_=updates[name](
                    arm["params"],arm["m"],arm["v"],*samples,step)
            if step==steps or step%max(1,steps//5)==0:
                for name,arm in arms.items():
                    metric=scores(arm["params"],validation,ticks,name)["mse"]
                    if metric<arm["val"]:
                        arm["best"],arm["step"],arm["val"]=arm["params"],step,metric
        record={"seed":int(seed),"arms":{}}
        for name,arm in arms.items():
            record["arms"][name]={"step":arm["step"],
                "validation_mse":arm["val"],
                "test":scores(arm["best"],test,ticks,name)}
            np.savez_compressed(folder/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray,arm["best"]))
        results.append(record)
    report={"experiment":"cellular_relevant_signal_modulation",
            "seeds":list(map(int,seeds)),"steps":steps,"ticks":ticks,
            "comparisons":list(model.MODES),"results":results,
            "caveat":"Source relevance is GIVEN as a cue mask, not discovered. "
                     "This is receiver-side message gating, not learned "
                     "sender-specific routing or general selective attention. "
                     "Plain mode allocates inactive gate parameters; active "
                     "capacity and compute therefore differ. No efficacy "
                     "claim before full training and replication."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/cellular-modulation")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=60)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.steps)["results"],indent=2))


if __name__=="__main__":
    main()
