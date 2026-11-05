"""Standardize cost comparisons for the same RGB world, batch and horizon.

JAX compilation, steady-state training, and warmed *full rollout* inference
are measured separately, with synchronization and repeated wall timings.
"""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from experiments.rgb_world_ladder_training import specification, objective
from src import rgb_world_ladder as world
from train import adamw

FAMILIES=("cnn","transport","hierarchical","nca","nca_blend","routing")


def block(tree):
    """Force JAX host-visible completion of every returned output."""
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf,"block_until_ready"):
            leaf.block_until_ready()


def distribution(values):
    values=np.asarray(values,np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Need nonempty finite timing samples")
    return {"n":len(values),"mean_ms":float(values.mean()*1000),
            "median_ms":float(np.median(values)*1000),
            "std_ms":float(values.std()*1000),
            "p05_ms":float(np.percentile(values,5)*1000),
            "p95_ms":float(np.percentile(values,95)*1000),
            "min_ms":float(values.min()*1000),
            "max_ms":float(values.max()*1000)}


def hardware():
    devices=jax.devices()
    return {
        "python":platform.python_version(),
        "os":platform.platform(),
        "machine":platform.machine(),
        "cpu_logical_count":os.cpu_count(),
        "jax":jax.__version__,"jaxlib":jaxlib.__version__,
        "numpy":np.__version__,
        "backend":jax.default_backend(),
        "device_count":len(devices),
        "device_kind":sorted(set(d.device_kind for d in devices)),
    }


def profile(name="cnn",seed=11,size=32,context=8,horizon=10,
            batch_size=2,width=8,ticks=2,train_steps=20,
            warmup=3,repetitions=20):
    if (context<2 or horizon<1 or size<20 or batch_size<1 or width<1
            or ticks<1 or train_steps<1 or warmup<0 or repetitions<2):
        raise ValueError("Invalid profiling dimensions/budget")
    params,rollout=specification(name,jax.random.PRNGKey(int(seed)),
                                 width,horizon,ticks)
    records=world.split("train",batch_size,context,horizon,size,2,"linear")
    images=jnp.asarray(np.stack([r["frames"] for r in records]))
    inputs=images[:,:context]
    m=jax.tree.map(jnp.zeros_like,params)
    v=jax.tree.map(jnp.zeros_like,params)

    def step(p,mom,second,batch,index):
        value,grads=jax.value_and_grad(lambda q:objective(
            q,batch,rollout,context,horizon))(p)
        norm=jnp.sqrt(sum((g*g).sum() for g in jax.tree.leaves(grads)))
        grads=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grads)
        updated,mom,second=adamw(p,grads,mom,second,index)
        return updated,mom,second,value

    # Compile explicitly and separately from steady-state run timers.
    stamp=time.perf_counter()
    trained=jax.jit(step).lower(params,m,v,images,jnp.asarray(1)).compile()
    train_compile_ms=(time.perf_counter()-stamp)*1000
    step_times=[]
    for n in range(1,train_steps+1):
        stamp=time.perf_counter()
        params,m,v,_=trained(params,m,v,images,jnp.asarray(n))
        block((params,m,v))
        step_times.append(time.perf_counter()-stamp)

    stamp=time.perf_counter()
    infer=jax.jit(lambda p,c:rollout(p,c)).lower(params,inputs).compile()
    infer_compile_ms=(time.perf_counter()-stamp)*1000
    for _ in range(warmup):
        block(infer(params,inputs))
    timings=[]
    for _ in range(repetitions):
        stamp=time.perf_counter()
        result=infer(params,inputs)
        block(result)
        timings.append(time.perf_counter()-stamp)

    return {"model":name,"seed":int(seed),
            "parameter_count":int(sum(x.size for x in jax.tree.leaves(params))),
            "compile_ms":{"training_step":train_compile_ms,
                          "inference_rollout":infer_compile_ms},
            "training_steady_step_ms":distribution(step_times),
            "training_steady_total_ms":float(sum(step_times)*1000),
            "inference_rollout_ms":distribution(timings),
            "workload":{"size":size,"context":context,"horizon":horizon,
                        "batch_size":batch_size,"width":width,"ticks":ticks,
                        "train_steps":train_steps,"warmup":warmup,
                        "repetitions":repetitions,
                        "timing_unit":"wall_clock_milliseconds",
                        "inference_unit":"full_autoregressive_rollout_per_batch",
                        "jit_compiled_before_repetitions":True,
                        "synchronization":"every JAX output blocked before stop"}}


def run(out="runs/comparable-compute/metrics.json",models=FAMILIES,
        seeds=(11,23,41),**settings):
    if not models or not seeds:
        raise ValueError("Need model families and seeds")
    settings.pop("out",None)
    rows=[profile(name,seed=int(seed),**settings)
          for name in models for seed in seeds]
    report={"protocol":"comparable-jax-compute-v1",
            "environment":hardware(),"results":rows,
            "limits":"Wall-clock samples on one host (compilation, steady "
              "training and full-rollout inference separated). These measure "
              "the selected JAX compilation and batch shape, not architecture "
              "FLOPs, energy, accuracy, or cross-machine efficiency. "
              "Ensure no competing system workload and compare at equal "
              "horizon/batch; do not compare training-only and inference-only "
              "numbers or claim identical trainable capacity."}
    path=Path(out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/comparable-compute/metrics.json")
    p.add_argument("--models",nargs="+",default=list(FAMILIES),
                   choices=FAMILIES)
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--width",type=int,default=8)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--train-steps",type=int,default=20)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repetitions",type=int,default=20)
    a=p.parse_args()
    result=run(a.out,a.models,a.seeds,size=a.size,context=a.context,
               horizon=a.horizon,batch_size=a.batch_size,width=a.width,
               ticks=a.ticks,train_steps=a.train_steps,
               warmup=a.warmup,repetitions=a.repetitions)
    print(json.dumps({"models":len(a.models),"seeds":len(a.seeds),
                      "results":len(result["results"])},indent=2))


if __name__=="__main__":
    main()
