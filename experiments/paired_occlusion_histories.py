"""Benchmark future contrast under identical last RGB, changed history."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import paired_occlusion_histories as world
from src import reappearance_scoring as events


def evaluate(predictor,records,context,horizon):
    contexts=np.stack([r["frames"][:context] for r in records])
    modes=world.controls(contexts)
    result={}
    for label,inputs in modes.items():
        forecast=np.asarray(predictor(inputs))
        if forecast.shape!=records[0]["frames"][context:context+horizon][None].repeat(
                len(records),axis=0).shape:
            raise ValueError("Incorrect predictive horizon/frame shape")
        scored=events.score(forecast,records,context)
        pairs=forecast.reshape((-1,2,horizon,*forecast.shape[2:]))
        predicted_diff=pairs[:,0]-pairs[:,1]
        truth=np.stack([r["frames"][context:context+horizon] for r in records])
        true_diff=truth.reshape(pairs.shape)[:,0]-truth.reshape(pairs.shape)[:,1]
        result[label]={"event_scores":scored,
            "future_contrast_mse":float(np.mean((predicted_diff-true_diff)**2)),
            "predicted_pair_difference":float(np.mean(np.abs(predicted_diff))),
            "truth_pair_difference":float(np.mean(np.abs(true_diff)))}
    return result


def run(out="runs/paired-occlusion/metrics.json",seeds=(11,23,41),
        context=8,horizon=10,size=32,checkpoint=None,model="cnn",
        width=8,ticks=2):
    if not seeds:
        raise ValueError("Need held-out seed pairs")
    records=[r for seed in seeds for r in world.pair(int(seed),context,horizon,size)]
    if checkpoint:
        from experiments.rgb_world_ladder_training import specification
        _,forward=specification(model,jax.random.PRNGKey(0),width,horizon,ticks)
        with np.load(checkpoint,allow_pickle=False) as item:
            params={key:jnp.asarray(item[key]) for key in item.files}
        predictor=lambda contexts:np.asarray(forward(params,jnp.asarray(contexts)))
        title=f"frozen_{model}"
    else:
        predictor=lambda contexts:np.repeat(contexts[:,-1:,],horizon,axis=1)
        title="persistence"
    compared=evaluate(predictor,records,context,horizon)
    observed=np.stack([r["frames"][:context] for r in records])
    fitted=np.stack([world.linear_history_forecast(x,horizon) for x in observed])
    result={"protocol":"history-identifiability-occlusion-v1",
            "seeds":list(map(int,seeds)),"context":context,"horizon":horizon,
            "size":size,"predictor":title,"comparisons":compared,
            "history_linear_baseline":events.score(fitted,records,context),
            "positive_control":events.score(np.stack([
                r["frames"][context:] for r in records]),records,context),
            "limits":"Paired last observed RGB frames are bit-identical. "
              "Only earlier motion reveals the expected reappearance side. "
              "The linear baseline knows the fixed colored-disc/stripe "
              "benchmark prior; no simulator future values enter it. "
              "No model has succeeded unless its held-out intact-history "
              "performance exceeds reset/shuffled and simple controls."}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/paired-occlusion/metrics.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--checkpoint")
    p.add_argument("--model",default="cnn")
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--size",type=int,default=32)
    a=p.parse_args()
    r=run(a.out,a.seeds,a.context,a.horizon,a.size,a.checkpoint,a.model)
    print(json.dumps({"mode":r["predictor"],"pairs":len(a.seeds)},indent=2))


if __name__=="__main__":
    main()
