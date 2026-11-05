"""Roadmap 35: validation-gated developmental curriculum, with fixed controls.

Train the same small observable-state predictor under three increasingly hard
synthetic regression distributions. Gates consult *validation* only; test
episodes are touched only after the last update. These stages are engineering
proxies, NOT certified visual-motion/occlusion/attention capabilities.
"""
import argparse
import json
from pathlib import Path
import numpy as np

STAGES = ("simple", "varied", "nonlinear")
MODES = ("gated", "fixed", "mixed")
# Exploratory toy thresholds, frozen before this protocol's test runs.
THRESHOLDS = (.055, .075)
CHECK_EVERY = 10
MIN_EXPOSURE = 10


def world(seed, stage, count=128):
    if stage not in range(len(STAGES)) or count < 1:
        raise ValueError("Invalid stage/count")
    rng=np.random.default_rng(int(seed))
    span=(.3, .8, 1.4)[stage]
    x=rng.uniform(-span,span,count)
    features=np.stack((np.ones(count),x,x*x,np.sin(3*x)),axis=-1)
    target=(.15 + .55*x + .28*x*x + .24*np.sin(3*x)
            + rng.normal(0,(.075,.12,.16)[stage],count))
    return features,target


def mse(weights, examples):
    x,y=examples
    return float(np.square(x@weights-y).mean())


def train_step(weights, example, lr=.15):
    x,y=example
    gradient=x.T@(x@weights-y)/len(x)
    norm=np.linalg.norm(gradient)
    if norm > 3.:
        gradient=3.*gradient/norm
    return weights-lr*gradient


def promote(stage, consecutive, exposed, validation_mse):
    """Two independent evaluation times under threshold before promotion."""
    if not 0 <= stage < len(STAGES):
        raise ValueError("Bad capability stage")
    if stage == len(STAGES)-1:
        return stage,consecutive,False
    ok=(exposed >= MIN_EXPOSURE and
        validation_mse <= THRESHOLDS[stage])
    streak=consecutive+1 if ok else 0
    if streak >= 2:
        return stage+1,0,True
    return stage,streak,False


def fit(seed, mode, steps, batch=12):
    if mode not in MODES or steps < CHECK_EVERY*2 or batch < 1:
        raise ValueError("Invalid matched training policy")
    # Pre-generated *training* streams are shared across methods.
    training=[world(31000+int(seed)*10+j,j,steps*batch)
              for j in range(len(STAGES))]
    validation=[world(61000+j,j,96) for j in range(len(STAGES))]
    testing=[world(91000+j,j,96) for j in range(len(STAGES))]
    rng=np.random.default_rng(int(seed)+22000)
    weights=np.zeros(4,float)
    stage=0
    streak=0
    exposed=0
    visits=[0]*len(STAGES)
    gates=[]
    curve=[]
    for step in range(steps):
        if mode=="fixed":
            selected=min(2,(step*len(STAGES))//steps)
        elif mode=="mixed":
            selected=int(rng.integers(len(STAGES)))
        else:
            selected=stage
        # Exactly 10% simple-stage replay in every arm regardless of policy.
        if (step+1) % 10 == 0:
            selected=0
        visits[selected]+=1
        if mode=="gated" and selected==stage:
            exposed+=1
        x,y=training[selected]
        subset=slice(step*batch,(step+1)*batch)
        weights=train_step(weights,(x[subset],y[subset]))
        if (step+1)%CHECK_EVERY==0:
            vals=[mse(weights,data) for data in validation]
            if mode=="gated":
                next_stage,streak,success=promote(stage,streak,exposed,vals[stage])
                if success:
                    gates.append({"update":step+1,"from":STAGES[stage],
                                  "to":STAGES[next_stage],
                                  "validation_mse":vals[stage]})
                    exposed=0
                stage=next_stage
            curve.append({"step":step+1,"active_stage":STAGES[stage],
                          "validation_mse":dict(zip(STAGES,vals))})
    final={name:mse(weights,data) for name,data in zip(STAGES,testing)}
    return {"seed":int(seed),"mode":mode,"updates":steps,"batch":batch,
            "examples_seen":steps*batch,"replay_simple_updates":steps//10,
            "stage_visits":dict(zip(STAGES,visits)),
            "promotions":gates,"validation_curve":curve,
            "test_mse_by_stage":final,"test_mean_stage_mse":float(np.mean(list(final.values()))),
            "weights":weights}


def run(out="runs/roadmap-35",seeds=(11,23,41),steps=120,batch=12):
    if not seeds:
        raise ValueError("Need at least one seed")
    results=[]
    folder=Path(out) if out is not None else None
    if folder is not None:
        folder.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        for mode in MODES:
            row=fit(int(seed),mode,steps,batch)
            if folder is not None:
                np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                    weights=row["weights"])
            results.append({k:v for k,v in row.items() if k!="weights"})
    summary={}
    for mode in MODES:
        rows=[r for r in results if r["mode"]==mode]
        summary[mode]={
            "heldout_mean_stage_mse":float(np.mean([r["test_mean_stage_mse"] for r in rows])),
            "per_stage_test_mse":{stage:float(np.mean([
                r["test_mse_by_stage"][stage] for r in rows])) for stage in STAGES},
            "promotion_counts":[len(r["promotions"]) for r in rows],
            "total_updates":steps}
    report={"roadmap_id":35,"seeds":list(map(int,seeds)),
            "curricula":list(MODES),"stage_labels":list(STAGES),
            "steps":steps,"batch_size":batch,
            "preregistered_exploratory_thresholds":dict(zip(STAGES[:2],THRESHOLDS)),
            "identical_update_and_example_budget":True,
            "simple_stage_replay_updates":steps//10,
            "summary":summary,"per_seed":results,
            "limits":"Three synthetic observed-feature regression distributions are "
                     "curriculum proxies, not the motion/occlusion/viewpoint/attention "
                     "capabilities in DEVELOPMENTAL_CAPABILITIES.md. Gates inspect "
                     "independent validation labels, never test. They may never promote; "
                     "no superiority is assumed. Match pixel/compute budgets and "
                     "freeze gates before confirmatory real RGB experiments."}
    if folder is not None:
        (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-35")
    p.add_argument("--steps",type=int,default=120)
    a=p.parse_args()
    print(json.dumps(run(a.out,steps=a.steps)["summary"],indent=2))


if __name__=="__main__":
    main()
