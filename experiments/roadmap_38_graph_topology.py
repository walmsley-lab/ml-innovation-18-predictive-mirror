"""Roadmap 38: train existing graph NCA with true, rewired, absent edges.

Input is GIVEN 2D joints and visibility, not inferred RGB pose. All arms
share network shape, initial weights, optimizer and observed sequences.
Hidden joint truth is evaluator-only. Cross-graph spectral interpretations
must not be inferred from simple predictive error differences alone.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from experiments import graph_skeleton_horizons as task
from experiments.graph_skeleton_nca import examples
from src import graph_skeleton_nca as nca, communication_graph_harmonics as harmonic
from train import adamw


def graphs(seed=11):
    true=harmonic.adjacency_from_edges(len(nca.JOINTS),nca.EDGES)
    rewired=harmonic.rewired(true,seed,swaps=20)
    if not np.array_equal(true.sum(0),rewired.sum(0)):
        raise AssertionError("Rewiring changed communication degree")
    def normalize(a):
        return jnp.asarray(a/np.maximum(a.sum(1,keepdims=True),1),jnp.float32)
    return {"skeleton":(normalize(true),True),
            "rewired":(normalize(rewired),True),
            "no_messages":(normalize(true),False)}


def run(out="runs/roadmap-38",seeds=(11,23,41),steps=30,
        context=8,horizon=5,batch=2,heldout=8,channels=8):
    if not seeds or steps<1 or context<2 or horizon<1 or context+horizon-1<12:
        raise ValueError("Need valid complete skeleton episode")
    task.check_disjoint(seeds,steps,batch,heldout)
    folder=Path(out) if out else None
    if folder:folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        options=graphs(int(seed)+100)
        initial=nca.init(jax.random.PRNGKey(int(seed)),channels)
        pairs=[task.train_ids(int(seed),t,batch) for t in range(1,steps+1)]
        for name,(adj,enabled) in options.items():
            params=initial
            first=jax.tree.map(jnp.zeros_like,params)
            second=jax.tree.map(jnp.zeros_like,params)
            @jax.jit
            def step(p,m,v,x,mask,i):
                value,grads=jax.value_and_grad(task.loss)(
                    p,x,mask,context,horizon,adj,enabled)
                norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grads)))
                grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grads)
                p,m,v=adamw(p,grads,m,v,i)
                return p,m,v
            started=time.perf_counter()
            for i,ids in enumerate(pairs,1):
                observed,mask,_=examples(ids,steps=context+horizon-1)
                params,first,second=step(params,first,second,
                                         jnp.asarray(observed),
                                         jnp.asarray(mask),i)
            jax.block_until_ready(jax.tree.leaves(params)[0])
            runtime=time.perf_counter()-started
            obs,mask,truth=examples(task.test_ids(int(seed),heldout),
                                    steps=context+horizon-1)
            scores={}
            prediction=np.asarray(task.rollout(params,jnp.asarray(obs[:,:context]),
                                               jnp.asarray(mask[:,:context]),
                                               horizon,adj,enabled))
            controls={"last_seen":task.last_seen(obs[:,:context],mask[:,:context],horizon),
                      "velocity":task.last_seen(obs[:,:context],mask[:,:context],
                                                 horizon,True)}
            for control_name,data in {"learned":prediction,**controls}.items():
                scores[control_name]=task.score(data,truth[:,context:],
                                        mask[:,context:],mask[:,context-1],
                                        (1,horizon))
            digest=None
            if folder:
                path=folder/f"seed-{seed}-{name}.npz"
                np.savez_compressed(path,**jax.tree.map(np.asarray,params))
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
            results.append({"seed":int(seed),"arm":name,"scores":scores,
                            "checkpoint_sha256":digest,
                            "train_seconds_including_jit":runtime,
                            "parameters":int(sum(x.size for x in jax.tree.leaves(params)))})
    report={"roadmap_id":38,"seeds":list(map(int,seeds)),"steps":steps,
            "context":context,"horizon":horizon,"rows":results,
            "representation":"provided normalized 2D joints and visibility, not pixels",
            "limits":"Rewires match degree, but cannot ensure matched diameter/connectedness. "
                     "No oracle occluded coordinates during training. Structural "
                     "graph vs zero messages cannot be fairly ranked against RGB "
                     "models without accounting for privileged joint inputs."}
    if folder:(folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-38")
    p.add_argument("--steps",type=int,default=30)
    a=p.parse_args()
    print(json.dumps({"rows":len(run(a.out,steps=a.steps)["rows"])}))


if __name__=="__main__":
    main()
