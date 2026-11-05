"""Run guaranteed, event-complete occlusion probes with frozen RGB models."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import guaranteed_reappearance as world, reappearance_scoring


def run(out="runs/guaranteed-reappearance/metrics.json",
        seeds=(11,23,41),context=4,horizon=10,size=32,mode="barrier",
        checkpoint=None,model="cnn",width=8,ticks=2):
    records=world.records(seeds,context,horizon,size,mode)
    contexts=np.stack([r["frames"][:context] for r in records])
    if checkpoint is None:
        # A useful positive control, not a model score.
        forecast=np.stack([r["frames"][context:] for r in records])
        evaluated="oracle_truth_positive_control"
    else:
        from experiments.rgb_world_ladder_training import specification
        _,predictor=specification(model,jax.random.PRNGKey(0),width,horizon,ticks)
        with np.load(checkpoint,allow_pickle=False) as f:
            params={k:jnp.asarray(f[k]) for k in f.files}
        forecast=np.asarray(predictor(params,jnp.asarray(contexts)))
        evaluated=model
    report=reappearance_scoring.score(forecast,records,context)
    report.update({"evaluated":evaluated,"context":context,
                   "horizon":horizon,"size":size,"mode":mode,
                   "seeds":list(map(int,seeds))})
    dest=Path(out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/guaranteed-reappearance/metrics.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--mode",choices=("barrier","camera_exit"),default="barrier")
    p.add_argument("--checkpoint")
    p.add_argument("--model",default="cnn")
    p.add_argument("--context",type=int,default=4)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--size",type=int,default=32)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.seeds,a.context,a.horizon,a.size,
                         a.mode,a.checkpoint,a.model)["results"],indent=2))


if __name__=="__main__":
    main()
