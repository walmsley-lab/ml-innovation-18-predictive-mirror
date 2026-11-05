"""Measure communication reach and a host-side training-memory proxy."""
import argparse
import json
import tracemalloc
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from src import communication_reach as reach, predictive_rgb_nca as nca


def memory_probe(seed=11,unrolls=(1,3,5),size=16,channels=4):
    """Traced host allocation and activation lower bound, not device HBM."""
    entries=[]
    for horizon in unrolls:
        key=jax.random.PRNGKey(int(seed))
        params=nca.init(key,channels)
        batch=jnp.full((1,3,size,size,3),.13)
        def loss(p):
            return jnp.mean(nca.rollout(p,batch,horizon,1)**2)
        jax.block_until_ready(jax.value_and_grad(loss)(params)[0])
        tracemalloc.start()
        value,grads=jax.value_and_grad(loss)(params)
        jax.block_until_ready(value)
        for x in jax.tree.leaves(grads):jax.block_until_ready(x)
        _,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        entries.append({"horizon":horizon,"host_python_traced_peak_bytes":int(peak),
                        "activation_lower_bound_bytes":int(
                            1*size*size*(channels+3)*4*horizon)})
    return {"method":"tracemalloc host Python allocations, excludes XLA device buffers",
            "rows":entries}


def run(out="runs/communication-reach/metrics.json",seeds=(11,23,41),
        size=24,context=4,horizon=3,ticks=2):
    if not seeds:
        raise ValueError("Need independent seeds")
    paths=[reach.compare(seed,size,context,horizon,ticks) for seed in seeds]
    report={"protocol":"communication-reach-audit-v1",
            "paths":paths,"memory":memory_probe(int(seeds[0])),
            "limits":"Report pathway-specific sensitivity and operation budgets; "
             "measured random-init reach is not validation of trained reach. "
             "Host tracemalloc is not GPU peak memory. Theoretical kr locality "
             "only applies to isolated local messages, not nonlocal warping. "
             "Displacement divided by measured reach is a hypothesis, not "
             "a universal threshold."}
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/communication-reach/metrics.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--size",type=int,default=24)
    a=p.parse_args()
    result=run(a.out,a.seeds,a.size)
    print(json.dumps({"seeds":len(result["paths"]),
                      "memory_runs":len(result["memory"]["rows"])}))


if __name__=="__main__":
    main()
