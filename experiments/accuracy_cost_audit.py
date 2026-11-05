"""Measure real JAX inference timing and optionally build accuracy-cost frontier."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca
from src import rgb_motion_transport as transport
from src.accuracy_cost_frontier import nondominated


def synchronize(output):
    for array in jax.tree.leaves(output):
        if hasattr(array,"block_until_ready"):
            array.block_until_ready()
    return output


def measure(fn,warmup=3,repeats=15):
    if warmup<1 or repeats<3:
        raise ValueError("Need repeated synchronized measurements")
    for _ in range(warmup):
        synchronize(fn())
    ms=[]
    for _ in range(repeats):
        t=time.perf_counter()
        synchronize(fn())
        ms.append((time.perf_counter()-t)*1000)
    return {"latency_ms":float(np.median(ms)),
            "p90_latency_ms":float(np.quantile(ms,.9)),
            "samples_ms":ms,"warmup":warmup,"repeats":repeats}


def measure_models(size=32,context=8,horizon=10,repeats=15):
    if size<16 or context<2 or horizon<1:
        raise ValueError("Invalid geometry")
    x=jnp.full((1,context,size,size,3),.075,jnp.float32)
    p=nca.init(jax.random.PRNGKey(11),channels=8)
    q=transport.init(jax.random.PRNGKey(11),width=8)
    functions={"rgb_nca":jax.jit(lambda:nca.rollout(p,x,horizon,2)),
               "rgb_transport":jax.jit(
                   lambda:transport.rollout(q,x,horizon,mode="transport"))}
    rows={}
    for name,fn in functions.items():
        rows[name]=measure(fn,repeats=repeats)
    return {"protocol":"random RGB inputs, 8-context + 10-forecast, no quality score",
            "device":str(jax.devices()[0]),"size":size,"context":context,
            "horizon":horizon,"timings":rows,
            "energy_j":None,"peak_memory_mib":None,
            "warning":"Inference timing only; no energy or physical-efficiency claim."}


def run(input_file=None,out="runs/accuracy-cost/metrics.json",repeats=15):
    if input_file:
        source=json.loads(Path(input_file).read_text())
        report={"comparison":nondominated(source["models"],
                    success_key=source.get("success_metric","success_rate"),
                    cost_keys=tuple(source.get("cost_metrics",["latency_ms"]))),
                "source":str(input_file),
                "warning":"Accuracy/cost measurements must share protocol and device"}
    else:
        report=measure_models(repeats=repeats)
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",help="JSON containing measured model rows")
    p.add_argument("--out",default="runs/accuracy-cost/metrics.json")
    p.add_argument("--repeats",type=int,default=15)
    a=p.parse_args()
    print(json.dumps(run(a.input,a.out,a.repeats),indent=2))


if __name__=="__main__":
    main()
