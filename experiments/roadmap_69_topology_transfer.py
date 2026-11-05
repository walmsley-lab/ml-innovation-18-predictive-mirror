"""69: transfer shared cellular coefficients between different toy graphs.

Keep pretrained update weights fixed while changing observed connectivity.
Test unseen target graph against stale-source topology, target-supervised
reference and shuffled target topology. All observations are node signals.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import roadmap_62_connectome_contract as contract
from experiments import roadmap_64_graph_dynamics as dynamics


def score(a,weights,episodes,horizon):
    initial=episodes[:,-horizon-1]
    future=episodes[:,-horizon:]
    forecast=dynamics.predict(a,weights,initial,horizon)
    return float(np.square(forecast-future).mean())


def run(out="runs/roadmap-69/metrics.json",seeds=(11,23,41),
        steps=12,horizon=4):
    if not seeds or steps<=horizon+2:
        raise ValueError("Need train/held-out trajectories and forecast interval")
    source,_=contract.toy(11)
    target,_=contract.toy(47)
    rows=[]
    for seed in seeds:
        source_train=dynamics.episodes(source,31000+int(seed),20,steps)
        target_train=dynamics.episodes(target,41000+int(seed),20,steps)
        source_test=dynamics.episodes(source,91000+int(seed),8,steps)
        target_test=dynamics.episodes(target,99000+int(seed),8,steps)
        pretrained=dynamics.fit(source,source_train)
        refit=dynamics.fit(target,target_train)
        null=dynamics.null_graph(target,int(seed)+400)
        cases={
            "source_in_domain":(source,pretrained,source_test),
            "transferred_weights_target_graph":(target,pretrained,target_test),
            "stale_source_topology":(source,pretrained,target_test),
            "target_supervised_reference":(target,refit,target_test),
            "target_shuffled_edges":(null,pretrained,target_test)}
        for name,(graph,weights,data) in cases.items():
            rows.append({"seed":int(seed),"arm":name,
                         "test_mse":score(graph,weights,data,horizon),
                         "source_graph_known":name=="source_in_domain",
                         "target_train_supervision":name=="target_supervised_reference",
                         "coefficients":weights.tolist()})
    report={"roadmap_id":69,"source_type":"synthetic",
            "source_graph":"toy seed 11","target_graph":"toy seed 47",
            "rows":rows,"forecast_horizon":horizon,
            "limits":"Toy graph-to-graph transfer tests topology-conditioned "
            "parameters, not species/developmental neural transfer. Target "
            "supervised fit has additional target labels and is a reference "
            "upper comparison, not equal-information training. No measured "
            "connectome or developmental neural data loaded."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-69/metrics.json")
    a=p.parse_args();print(json.dumps(run(a.out),indent=2))
