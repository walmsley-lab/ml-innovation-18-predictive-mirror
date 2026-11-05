"""Roadmap 35 RGB bridge: validation-gated training for the real RGB NCA.

Staged visual domains (visible movement, occlusion, camera shift) are proxy
competence tests, NOT certifications of full object permanence or ego-motion.
Every arm uses the same NCA, initialization, update count and held-out scenes.
No test episode participates in stage transitions.
"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import rgb_motion_world as worlds, rgb_benchmark as metric
from src import predictive_rgb_nca as nca

DOMAINS=("motion","occlusion","camera")
ARMS=("gated","fixed","mixed")


def samples(start,count,domain,context=4,horizon=1,size=20):
    if domain not in DOMAINS or count<1:
        raise ValueError("Invalid RGB domain")
    return worlds.batch(
        range(start,start+count),steps=context+horizon,size=size,
        balls=1,occlusion=(domain=="occlusion"),
        camera_motion=(domain=="camera"))


def evaluate(params,frames,context,ticks):
    pred=np.asarray(nca.rollout(params,jnp.asarray(frames[:,:context]),1,ticks))
    report=metric.score(pred,frames,context,(1,))["1"]
    # Motion-region MSE is defined only when some target pixels change.
    return {"rgb_mse":report["mse"],"motion_mse":report["motion_mse"],
            "motion_pixels":report["motion_pixels"]}


def baseline(frames,context):
    p=metric.controls(frames[:,:context],1)
    return {name:metric.score(value,frames,context,(1,))["1"]
            for name,value in p.items()}


def step(params,batch,context,ticks,rate=.02):
    objective=lambda p:nca.loss(p,jnp.asarray(batch),context,1,ticks)
    _,grads=jax.value_and_grad(objective)(params)
    norm=jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
    return jax.tree.map(lambda p,g:p-rate*g/jnp.maximum(norm,1.),params,grads)


def gate(stage,streak,exposures,score,base,minimum=6):
    """Require two successful validation checks, no access to future test."""
    if not 0<=stage<len(DOMAINS):
        raise ValueError("Invalid stage")
    if stage==len(DOMAINS)-1:
        return stage,streak,False
    if score is None or base is None or base<=0:
        return stage,0,False
    # Exploratory: at least 1% change-region gain over persistence.
    passed=exposures>=minimum and score<=.99*base
    streak=streak+1 if passed else 0
    return (stage+1,0,True) if streak>=2 else (stage,streak,False)


def learn(seed,arm,steps=30,context=4,size=20,channels=4,ticks=1,
          validate_every=5,train_per_domain=None):
    if (arm not in ARMS or steps<2*validate_every or validate_every<1
            or context<2 or size<16 or channels<1 or ticks<1):
        raise ValueError("Invalid visual training setup")
    train_per_domain=steps if train_per_domain is None else train_per_domain
    if train_per_domain<steps:
        raise ValueError("Need enough independent training episode IDs")
    training=[samples(31000+int(seed)*1000+j*200,train_per_domain,name,
                      context,1,size) for j,name in enumerate(DOMAINS)]
    validation=[samples(61000+j*100,4,name,context,1,size)
                for j,name in enumerate(DOMAINS)]
    testing=[samples(91000+j*100,4,name,context,1,size)
             for j,name in enumerate(DOMAINS)]
    baselines=[baseline(frames,context) for frames in validation]
    initial=nca.init(jax.random.PRNGKey(int(seed)),channels)
    params=initial
    rng=np.random.default_rng(int(seed)+1770)
    stage=0
    streak=0
    exposures=0
    visits=[0,0,0]
    promotions=[]
    trace=[]
    for index in range(steps):
        if arm=="fixed":
            chosen=min(len(DOMAINS)-1,index*len(DOMAINS)//steps)
        elif arm=="mixed":
            chosen=int(rng.integers(len(DOMAINS)))
        else:
            chosen=stage
        if (index+1)%10==0:
            chosen=0  # same 10% simple-motion replay locations
        visits[chosen]+=1
        exposures+=int(arm=="gated" and chosen==stage)
        # Paired on-demand episode index: same step/seed/domain gives same RGB
        # for each arm even if the selected domain differs.
        batch=training[chosen][index:index+1]
        params=step(params,batch,context,ticks)
        if (index+1)%validate_every==0:
            measured=[evaluate(params,x,context,ticks) for x in validation]
            if arm=="gated":
                current=measured[stage]["motion_mse"]
                prior=baselines[stage]["persistence"]["motion_mse"]
                later,streak,passed=gate(stage,streak,exposures,current,prior)
                if passed:
                    promotions.append({"update":index+1,
                                       "from":DOMAINS[stage],"to":DOMAINS[later],
                                       "validation_motion_mse":current,
                                       "persistence_motion_mse":prior})
                    exposures=0
                stage=later
            trace.append({"step":index+1,"active_stage":DOMAINS[stage],
                          "validation":dict(zip(DOMAINS,measured))})
    final={name:evaluate(params,test,context,ticks)
           for name,test in zip(DOMAINS,testing)}
    controls={name:baseline(test,context) for name,test in zip(DOMAINS,testing)}
    return {"seed":int(seed),"arm":arm,"steps":steps,
            "stage_visits":dict(zip(DOMAINS,visits)),
            "promotions":promotions,"trace":trace,
            "test":final,"test_controls":controls,"parameters":params}


def run(out="runs/roadmap-35-rgb",seeds=(11,23,41),
        steps=30,context=4,size=20,channels=4,ticks=1,validate_every=5):
    if not seeds:
        raise ValueError("Need independent seeds")
    folder=Path(out) if out is not None else None
    if folder is not None:
        folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        for arm in ARMS:
            r=learn(int(seed),arm,steps,context,size,channels,ticks,
                    validate_every)
            if folder is not None:
                np.savez_compressed(folder/f"seed-{seed}-{arm}.npz",
                                    **jax.tree.map(np.asarray,r["parameters"]))
            rows.append({k:v for k,v in r.items() if k!="parameters"})
    summary={}
    for arm in ARMS:
        selected=[r for r in rows if r["arm"]==arm]
        summary[arm]={
            "mean_test_rgb_mse_by_domain":{
                name:float(np.mean([r["test"][name]["rgb_mse"]
                                    for r in selected])) for name in DOMAINS},
            "promotion_counts":[len(r["promotions"]) for r in selected],
            "per_seed_steps":[r["steps"] for r in selected]}
    report={
        "roadmap_id":35,"protocol":"rgb-nca-validation-gating-v1",
        "seeds":list(map(int,seeds)),"steps":steps,"context":context,
        "size":size,"channels":channels,"ticks":ticks,
        "validate_every":validate_every,
        "gate":"two validation checks with >=1% motion-MSE gain over persistence",
        "gate_threshold_status":"exploratory, not confirmatory capability certification",
        "equal_update_counts":True,"test_seed_start":91000,
        "summary":summary,"per_seed":rows,
        "limits":"Actual RGB NCA and independent pixel-based worlds, but motion, "
                 "occlusion and camera stages are domain proxies. No object- "
                 "permanence or ego-motion identification is certified. "
                 "Calibration on validation only; test scenes are untouched "
                 "until training ends. Effective data diversity differs by arm."}
    if folder is not None:
        (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-35-rgb")
    p.add_argument("--steps",type=int,default=30)
    a=p.parse_args()
    print(json.dumps(run(a.out,steps=a.steps)["summary"],indent=2))


if __name__=="__main__":
    main()
