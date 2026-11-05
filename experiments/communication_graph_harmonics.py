"""Compare layout-grid and actual communication graph harmonic effects."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import communication_graph_harmonics as h
from src.graph_skeleton_nca import EDGES,JOINTS


def run(out="runs/communication-graph-harmonics/metrics.json",
        seeds=(11,23,41),steps=5,rank=3):
    if not seeds or steps<1:
        raise ValueError("Need seeds/steps")
    # 12 skeleton nodes can be drawn as 3x4 pixels for comparison, but
    # the skeleton message edges are NOT the rectangular-grid edges.
    actual=h.adjacency_from_edges(len(JOINTS),EDGES)
    grid=h.lattice_adjacency(3,4)
    results=[]
    for seed in seeds:
        rng=np.random.default_rng(int(seed))
        initial=rng.normal(0,1,(len(JOINTS),3))
        random_graph=h.rewired(actual,int(seed)+11,swaps=30)
        cases={}
        for name,graph in (("true_communication_graph",actual),
                           ("display_grid",grid),
                           ("degree_preserving_rewire",random_graph)):
            metrics=h.interventions(graph,initial,steps,rank,int(seed)+27)
            cases[name]={k:v for k,v in metrics.items()
                         if not isinstance(v,np.ndarray)}
            cases[name]["graph_edges"]=int(graph.sum()//2)
            cases[name]["degree_sequence"]=graph.sum(axis=0).astype(int).tolist()
        results.append({"seed":int(seed),"cases":cases})
    report={"protocol":"communication-vs-layout-harmonics-v1",
            "seeds":list(map(int,seeds)),"rank":rank,"steps":steps,
            "rows":results,
            "limits":"These are spectra and causal interventions in a "
              "known linear graph diffusion reference, NOT yet a trained "
              "graph NCA. Display-grid Laplacian is a different adjacency; "
              "rewiring preserves degrees but may alter connectedness. "
              "Matched null removes equal L2 activity, not necessarily "
              "equal graph-frequency energy. Positive perturbation response "
              "does not imply better video prediction."}
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/communication-graph-harmonics/metrics.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--rank",type=int,default=3)
    a=p.parse_args()
    report=run(a.out,a.seeds,rank=a.rank)
    print(json.dumps({"seeds":report["seeds"],"rank":report["rank"]}))


if __name__=="__main__":
    main()
