"""Calibrate causal effect sizes on a frozen NCA, horizon by horizon."""
import argparse
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from src import functional_state_interventions as intervene
from src import rgb_motion_distribution as world
from src import forecast_diagnostics as diag


def run(checkpoint,out="runs/functional-state",seed_base=71000,
        episodes=6,context=8,horizon=10,ticks=2,output="residual"):
    if episodes<2 or context<2 or horizon<1:
        raise ValueError("Need distinct scenes and context")
    with np.load(checkpoint,allow_pickle=False) as data:
        params={k:jnp.asarray(data[k]) for k in data.files}
    frames,_=world.scenes(seed_base,episodes,context+horizon,32)
    obs=jnp.asarray(frames[:,:context])
    forecasts={mode:np.asarray(intervene.forecast(
        params,obs,horizon,ticks,mode,output)) for mode in intervene.MODES}
    intact=forecasts["intact"]
    truth=frames[:,context:context+horizon]
    leads=tuple(sorted({1,horizon,*[x for x in (3,5) if x<=horizon]}))
    report=diag.collapse_report(intact,leads,positive_control={
        "reset_hidden":forecasts["reset_hidden"],
        "shock_image":forecasts["shock_image"]})
    results={}
    for mode,pred in forecasts.items():
        results[mode]={
            "per_lead_mse":[float(np.mean((pred[:,i]-truth[:,i])**2))
                            for i in range(horizon)],
            "mean_forecast_shift":float(np.mean(np.abs(pred-intact)))
        }
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"forecast-comparison.npz",**forecasts)
    metadata={"checkpoint":str(checkpoint),"episode_seed_base":seed_base,
              "episodes":episodes,"context":context,"horizon":horizon,"output_head":output,
              "positive_control_interpretation":
                "If no known control moves a lead, a targeted null is uninterpretable.",
              "measurement":report,"arms":results,
              "no_claim":"Forecast sensitivity != useful learned object mechanism"}
    (folder/"metrics.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",default="runs/functional-state")
    p.add_argument("--episodes",type=int,default=6)
    p.add_argument("--output",choices=("residual","blend"),default="residual")
    a=p.parse_args()
    result=run(a.checkpoint,a.out,episodes=a.episodes,output=a.output)
    print(json.dumps(result["measurement"],indent=2))


if __name__=="__main__":
    main()
