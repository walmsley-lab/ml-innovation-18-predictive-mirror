"""68: same observed states and coefficients across graph, lattice, particles.

This is a matched INPUT-SIGNAL and parameter-count comparison in a
synthetic graph-teacher world, not real electrical/hardware substrate tests.
Particle locations are a disclosed artificial layout (not RGB-detected).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import roadmap_62_connectome_contract as contract
from experiments import roadmap_63_retinotopy as registration
from experiments import roadmap_64_graph_dynamics as dyn
from src import communication_graph_harmonics as graph_ops


def choices(n=12,scale=.3):
    if n!=12:
        raise ValueError("Synthetic cross-substrate fixture uses 12 nodes")
    reference,_=contract.toy(11,n)
    lattice=graph_ops.lattice_adjacency(3,4)
    positions=registration.toy_positions(n)
    distance=np.sum((positions[:,None]-positions[None,:])**2,axis=-1)
    particles=np.exp(-distance/(scale*scale))
    np.fill_diagonal(particles,0.)
    return {"connectome":reference,"grid_3x4":lattice,
            "particle_proximity":particles,"uncoupled":np.zeros_like(reference)}


def communication_cost(a,steps):
    nonzero=int(np.count_nonzero(a))
    return {"directed_message_edges":nonzero,
            "edge_reads_proxy":nonzero*steps,
            "hardware_energy_joules":None,
            "peripheral_overhead_bytes":None}


def run(out="runs/roadmap-68/metrics.json",seeds=(11,23,41),horizon=4):
    graph=choices()
    teacher=graph["connectome"]
    rows=[]
    for seed in seeds:
        train=dyn.episodes(teacher,31000+int(seed),20,12)
        test=dyn.episodes(teacher,91000+int(seed),8,12)
        anchor=test[:,7]
        truth=test[:,8:8+horizon]
        for name,a in graph.items():
            weights=dyn.fit(a,train)
            forecast=dyn.predict(a,weights,anchor,horizon)
            rows.append({"seed":int(seed),"substrate":name,
                         "test_mse":float(np.square(forecast-truth).mean()),
                         "parameters":len(weights),
                         "communication":communication_cost(a,horizon)})
    report={"roadmap_id":68,"source_type":"synthetic",
            "source":"12-node fixed toy graph as teacher; synthetic ring node positions",
            "seeds":list(map(int,seeds)),"rows":rows,"horizon":horizon,
            "limitations":"Equal observed-state information and three trainable "
             "coefficients; unequal edge count, geometry and communication "
             "volume are reported. Original graph is privileged by the "
             "teacher-generation process. These are simulated graph operators "
             "on a conventional CPU, NOT measured particle device or "
             "neuromorphic hardware energy."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-68/metrics.json")
    a=p.parse_args();print(json.dumps(run(a.out),indent=2))
