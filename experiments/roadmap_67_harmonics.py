"""67: causal graph-harmonic interventions on a frozen graph predictor.

Spectral modes are defined on an EXPLICIT SYMMETRIZED connection graph.
They are not traveling brain waves. Compare low-mode removal against equal
signal-energy random projections and a no-lesion sham.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src import communication_graph_harmonics as harmonic
from experiments import roadmap_62_connectome_contract as contract
from experiments import roadmap_64_graph_dynamics as dynamics


def basis(graph,rank=3):
    # Weighted undirected Laplacian is constructed *only for analysis*.
    symmetric=.5*(graph+graph.T)
    eigenvalues,components=harmonic.eigenbasis(symmetric,rank)
    return eigenvalues,components


def modes(state,components,seed):
    value=np.asarray(state,float)
    if value.ndim!=1 or components.shape[1]!=len(value):
        raise ValueError("Node/graph mode mismatch")
    centered=value-value.mean()
    low=components.T@(components@centered)
    rng=np.random.default_rng(seed)
    raw=rng.normal(size=(len(value),len(components)))
    raw-=raw.mean(0)
    q,_=np.linalg.qr(raw,mode="reduced")
    random_projected=q@(q.T@centered)
    magnitude=np.linalg.norm(low)
    control=random_projected*(
        magnitude/max(np.linalg.norm(random_projected),1e-12))
    return {"low":low,"matched_random":control,"sham":np.zeros_like(value)}


def run(out="runs/roadmap-67/metrics.json",seeds=(11,23,41),
        matrix=None,metadata=None,rank=3,horizon=4):
    if bool(matrix)!=bool(metadata):
        raise ValueError("Matrix and provenance must be paired")
    if matrix:
        a,_,meta=contract.load(matrix,metadata)
    else:
        a,_=contract.toy()
        meta={"source_type":"synthetic","source_id":"toy"}
    eigenvalues,components=basis(a,rank)
    rows=[]
    for seed in seeds:
        train=dynamics.episodes(a,31000+int(seed),20,12)
        evaluation=dynamics.episodes(a,91000+int(seed),4,12)
        weights=dynamics.fit(a,train)
        for sequence in evaluation:
            last=sequence[7]
            target=sequence[8:8+horizon]
            unperturbed=dynamics.predict(a,weights,last[None],horizon)[0]
            original=float(np.square(unperturbed-target).mean())
            interventions=modes(last,components,int(seed))
            for name,removed in interventions.items():
                altered=dynamics.predict(a,weights,(last-removed)[None],horizon)[0]
                rows.append({"seed":int(seed),"arm":name,
                             "removed_energy":float(removed@removed),
                             "accuracy_damage_mse":float(np.square(altered-target).mean()-original),
                             "forecast_delta_l2":float(np.linalg.norm(altered-unperturbed))})
    result={"roadmap_id":67,"source_type":meta["source_type"],
            "source_id":meta["source_id"],"graph_symmetrized_for_spectrum":True,
            "eigenvalues":eigenvalues.tolist(),"mode_count":len(components),
            "rows":rows,"limits":"Combinatorial graph Laplacian on undirected "
            "weighted symmetrization, not a physical propagation wave. "
            "Causal mode effects are in a trained *synthetic teacher* model; "
            "no biological cortical lesion or neural-agreement evidence. "
            "Matched control removal energy, not necessarily graph frequency."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-67/metrics.json")
    a=p.parse_args();print(json.dumps(run(a.out),indent=2))
