"""65: lesion trained graph dynamics with structurally matched controls.

A frozen predictor trained on the intact graph is intervened on by deleting
one node's communications. The target remains the intact simulated system:
this measures a counterfactual model dependency, NOT empirical neural lesion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import roadmap_62_connectome_contract as contract
from experiments import roadmap_64_graph_dynamics as dynamics


def lesion(a,node):
    x=np.asarray(a,float).copy()
    if not 0<=node<len(x):
        raise ValueError("Invalid lesion node")
    x[node,:]=0.
    x[:,node]=0.
    return x


def nodes(a,seed):
    degree=(a>0).sum(0)+(a>0).sum(1)
    hub=int(np.argmax(degree))
    candidates=[i for i in range(len(a)) if i!=hub]
    matched=min(candidates,key=lambda i:(abs(degree[i]-degree[hub]),i))
    random=int(np.random.default_rng(int(seed)).choice(candidates))
    return {"highest_degree":hub,"nearest_degree":matched,"random":random},degree


def run(out="runs/roadmap-65/metrics.json",seeds=(11,23,41),
        matrix=None,metadata=None,horizon=4):
    if bool(matrix)!=bool(metadata):
        raise ValueError("Matrix and provenance must be paired")
    if matrix:
        a,_,meta=contract.load(matrix,metadata)
    else:
        a,_=contract.toy()
        meta={"source_type":"synthetic","source_id":"toy"}
    rows=[]
    for seed in seeds:
        train=dynamics.episodes(a,31000+int(seed),16,12)
        test=dynamics.episodes(a,91000+int(seed),8,12)
        learned=dynamics.fit(a,train)
        start=test[:,7]
        target=test[:,8:8+horizon]
        intact=dynamics.predict(a,learned,start,horizon)
        reference=float(np.square(intact-target).mean())
        altered={"sham":a}
        selected,degree=nodes(a,int(seed))
        for name,node in selected.items():
            altered[name]=lesion(a,node)
        for name,graph in altered.items():
            forecast=dynamics.predict(graph,learned,start,horizon)
            rows.append({"seed":int(seed),"arm":name,
                         "intervened_node":selected.get(name),
                         "intervened_degree":int(degree[selected[name]])
                              if name in selected else 0,
                         "test_mse":float(np.square(forecast-target).mean()),
                         "damage_delta_mse":float(np.square(forecast-target).mean()-reference),
                         "output_effect_l2":float(np.linalg.norm(forecast-intact))})
    report={"roadmap_id":65,"source_type":meta["source_type"],
            "source_id":meta["source_id"],"rows":rows,
            "limitations":"Frozen *simulated* connectome dynamics. Target is the "
             "intact synthetic trajectory; damage reflects broken model "
             "computation, not animal lesion data. Nearest-degree control "
             "can have a nonzero degree mismatch; report actual degree."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-65/metrics.json")
    a=p.parse_args();print(json.dumps(run(a.out),indent=2))
