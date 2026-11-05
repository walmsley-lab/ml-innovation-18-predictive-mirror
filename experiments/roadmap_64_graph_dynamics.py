"""64: learned local dynamics constrained by an explicitly supplied graph.

Teacher dynamics are synthetic and constructed from this graph. Recovering
their transition is an engineering identifiability test, not evidence that
an animal connectome helps visual prediction.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import roadmap_62_connectome_contract as contract


def normalize(a):
    x=contract.validate(a,np.arange(len(a)))
    return x/np.maximum(x.sum(1,keepdims=True),1)


def null_graph(a,seed):
    """Shuffle outgoing targets/weights within each row; match row budgets."""
    rng=np.random.default_rng(seed)
    result=np.zeros_like(a)
    n=len(a)
    for i,row in enumerate(a):
        choices=np.arange(n)
        choices=choices[choices!=i]
        existing=row[row>0]
        destinations=rng.choice(choices,size=len(existing),replace=False)
        result[i,destinations]=rng.permutation(existing)
    assert np.allclose(result.sum(1),a.sum(1))
    return result


def episodes(a,start,count,steps=12):
    if count<2 or steps<4:
        raise ValueError("Need multiple independent sequences")
    rng=np.random.default_rng(start)
    p=normalize(a)
    result=[]
    for _ in range(count):
        state=rng.normal(size=len(a))
        series=[state.copy()]
        for t in range(steps-1):
            state=.65*state+.3*p@state+.03*np.sin(t*.7)
            series.append(state.copy())
        result.append(series)
    return np.asarray(result)


def fit(a,sequences,ridge=1e-3):
    p=normalize(a)
    current=sequences[:,:-1]
    targets=sequences[:,1:]
    messages=np.einsum("ij,btj->bti",p,current)
    design=np.stack((current,messages,np.ones_like(current)),axis=-1).reshape(-1,3)
    y=targets.reshape(-1)
    return np.linalg.solve(design.T@design+ridge*np.eye(3),design.T@y)


def predict(a,coefficients,last,horizon):
    if horizon<1 or len(coefficients)!=3:
        raise ValueError("Valid horizon and three shared weights required")
    p=normalize(a)
    state=np.asarray(last,float).copy()
    output=[]
    for _ in range(horizon):
        state=coefficients[0]*state+coefficients[1]*(state@p.T)+coefficients[2]
        output.append(state.copy())
    return np.stack(output,axis=1)


def run(out="runs/roadmap-64/metrics.json",matrix=None,metadata=None,
        seeds=(11,23,41),steps=12,horizon=4):
    if bool(matrix)!=bool(metadata):
        raise ValueError("Matrix and metadata must be paired")
    if matrix:
        a,_,info=contract.load(matrix,metadata)
    else:
        a,_=contract.toy()
        info={"source_type":"synthetic","source_id":"toy-reference"}
    if not seeds or steps<=horizon+2:
        raise ValueError("Need history before forecast")
    rows=[]
    for seed in seeds:
        train=episodes(a,31000+int(seed),20,steps)
        test=episodes(a,91000+int(seed),8,steps)
        for arm,graph in (("supplied",a),("row_budget_shuffle",null_graph(a,int(seed)+3)),
                          ("no_messages",np.zeros_like(a))):
            coefficients=fit(graph,train)
            pred=predict(graph,coefficients,test[:,steps-horizon-1],horizon)
            truth=test[:,steps-horizon:]
            rows.append({"seed":int(seed),"arm":arm,
                         "test_mse":float(np.square(pred-truth).mean()),
                         "coefficients":coefficients.tolist(),
                         "row_strength_mean":float(np.mean(graph.sum(1)))})
    result={"roadmap_id":64,"source_type":info["source_type"],
            "source_id":info["source_id"],"rows":rows,"horizon":horizon,
            "limitations":"Synthetic teacher generated from the supplied graph; "
            "in-distribution graph advantage is expected and cannot prove "
            "biological predictive usefulness. Shuffling matches outgoing "
            "strength but not incoming strength or topology geometry. "
            "Three trainable shared coefficients per arm; future test "
            "states are scoring-only."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--matrix");p.add_argument("--metadata")
    p.add_argument("--out",default="runs/roadmap-64/metrics.json")
    a=p.parse_args();print(json.dumps(run(a.out,a.matrix,a.metadata),indent=2))
