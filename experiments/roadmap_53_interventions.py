"""53: causal intervention on a frozen attention-routing computation.

Intervene on selected routing windows after using only observed RGB context.
For each window removal compare a randomly removed window. Same nominal
window count; record actual unique pixel area to expose unequal overlap.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import roadmap_46_attention as attention
from src import rgb_world_ladder as world, rgb_benchmark as bench


def forecast(context,selected,horizon=3):
    controls=bench.controls(context[None],horizon)
    hold=controls["persistence"][0]
    extrapolate=controls["constant_velocity"][0]
    return np.where(selected[None,:,:,None],extrapolate,hold)


def evaluate(frames,context=5,horizon=3,seed=11,width=6,count=4):
    observed=frames[:context]
    origins=attention.choose(observed,"surprise",count,width,seed)
    rng=np.random.default_rng(int(seed)+911)
    null_index=int(rng.integers(len(origins)))
    conditions={
        "intact":origins,
        "high_priority_cut":origins[1:],
        "matched_random_window_cut":[p for i,p in enumerate(origins) if i!=null_index],
        "no_attention":[]}
    outputs={key:forecast(observed,attention.mask(positions,len(observed[0]),width),
                          horizon) for key,positions in conditions.items()}
    truth=frames[context:context+horizon]
    before=outputs["intact"]
    return {name:{"mse":float(np.square(pred-truth).mean()),
                  "causal_output_delta":float(np.square(pred-before).mean()),
                  "window_count":len(conditions[name]),
                  "selected_pixels":int(attention.mask(conditions[name],
                                               len(observed[0]),width).sum())}
            for name,pred in outputs.items()}


def run(out="runs/roadmap-53/metrics.json",seeds=range(91000,91032),
        domains=("linear","accelerating","occlusion_long","camera_pan")):
    rows=[]
    for domain in domains:
        for seed in seeds:
            frames=world.episode(int(seed),8,24,domain=domain)["frames"]
            rows.append({"seed":int(seed),"domain":domain,
                         "interventions":evaluate(frames,seed=int(seed))})
    result={"roadmap_id":53,"rows":rows,
            "limits":"Intervenes on hand-crafted routing windows, not hidden "
            "synapses or trained NCA networks. Random null matches removed "
            "window count but can differ in area due to overlap; report both. "
            "Output difference alone is not functional importance: "
            "accuracy difference is measured separately."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/roadmap-53/metrics.json")
    print(json.dumps({"episodes":len(run(p.parse_args().out)["rows"])}))
