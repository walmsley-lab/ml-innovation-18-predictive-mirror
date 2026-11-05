"""Roadmap 42: constrained cellular rule versus physical I/O budget.

Compare a fixed two-state circuit update (full/quantized/local ablations)
on the SAME RGB episodes. Report actual CPU bytes/time and separately the
optimistic physical state+ADC/DAC transaction counts. Never invent joules
or assume simulated 4-bit values use packed physical memory.
"""
import argparse
import json
import platform
from pathlib import Path
import time
import numpy as np
from src import rgb_circuit_cell as circuit, rgb_benchmark as bench


def accounting(size,context,horizon,ticks,batch=1):
    if min(size,context,horizon,ticks,batch)<1:
        raise ValueError("Positive spatial/time budget required")
    sites=size*size*3*batch
    time_steps=context+horizon
    return {
        "state_scalar_count":2*sites,
        "numpy_state_bytes_float32":2*sites*4,
        "ideal_packed_4bit_state_bytes_lower_bound":int(np.ceil(2*sites*4/8)),
        "sensor_samples_lower_bound":sites*time_steps,
        "rgb_readout_samples_lower_bound":sites*horizon,
        "rgb_input_dac_transactions_lower_bound":sites*context,
        "rgb_output_adc_transactions_lower_bound":sites*horizon,
        "cell_updates":sites*time_steps*ticks,
        "physical_energy_joules":None,
        "required_peripheral_area":None,
        "unknowns":["ADC energy/precision","DAC energy/precision",
                    "state retention and noise","routing/fanout",
                    "readout time and bandwidth","actual physical device cost"],
    }


def latency(call,warmup=1,repeats=3):
    if repeats<2 or warmup<0:
        raise ValueError("Need at least two latency repetitions")
    for _ in range(warmup):call()
    samples=[]
    for _ in range(repeats):
        began=time.perf_counter()
        call()
        samples.append((time.perf_counter()-began)*1000)
    return {"median_ms":float(np.median(samples)),
            "p95_ms":float(np.percentile(samples,95)),
            "repeats":repeats}


def run(out="runs/roadmap-42/metrics.json",sizes=(16,24),
        modes=("full","quantized_4bit","no_coupling","no_slow_memory"),
        seed=11,train_count=5,test_count=5,context=4,horizon=3,ticks=1,
        repeats=3):
    if not sizes or not modes or min(train_count,test_count,context,horizon,ticks)<1:
        raise ValueError("Invalid hardware campaign")
    rows=[]
    for size in sizes:
        train=bench.frames(bench.scenes(31000+seed,train_count,context,horizon,size))
        test=bench.frames(bench.scenes(91000+seed,test_count,context,horizon,size))
        controls=bench.controls(test[:,:context],horizon)
        baseline={k:bench.score(v,test,context,(1,horizon))
                  for k,v in controls.items()}
        for mode in modes:
            head=circuit.fit_head(train,context,mode,ticks)
            def forecast():
                return circuit.rollout(test[:,:context],head,horizon,mode,ticks)
            predicted=forecast()
            scores=bench.score(predicted,test,context,(1,horizon))
            hardware=accounting(size,context,horizon,ticks,len(test))
            hardware["actual_state_dtype"]="float32 even in quantized_4bit mode"
            rows.append({"size":int(size),"mode":mode,
                         "scores":scores,"baselines":baseline,
                         "cpu_latency":latency(forecast,repeats=repeats),
                         "head_coefficients":head.tolist(),"hardware":hardware})
    report={"roadmap_id":42,"rows":rows,
            "host":platform.platform(),"python":platform.python_version(),
            "context":context,"horizon":horizon,"ticks":ticks,
            "limits":"Hardware state packing is a lower bound, not measured RAM saving. "
             "CPU timing is not electrical power or physical latency. "
             "Quantization and nonlinear circuit recurrence are synthetic. "
             "ADC/DAC, peripheral, interconnect and actual device energy are "
             "unknown; do not make neuromorphic advantage claims."}
    if out:
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-42/metrics.json")
    p.add_argument("--repeats",type=int,default=3)
    a=p.parse_args()
    print(json.dumps({"runs":len(run(a.out,repeats=a.repeats)["rows"])}))


if __name__=="__main__":
    main()
