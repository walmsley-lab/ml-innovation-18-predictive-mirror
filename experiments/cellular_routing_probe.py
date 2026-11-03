"""Frozen routing lesions: learned, uniform, shuffled, and key knockout."""
import argparse
import hashlib
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from src.cellular_routing_probe import analyze
from experiments.cellular_references import sample


def run(checkpoint, out="runs/cellular-routing-probe/metrics.json",
        seed=87000, episodes=32, size=10, ticks=4):
    if episodes < 1 or size < 8 or ticks < 1:
        raise ValueError("Invalid routing evaluation")
    with np.load(checkpoint,allow_pickle=False) as file:
        params={k:jnp.asarray(file[k]) for k in file.files}
    examples=sample(seed,episodes,size,low=5,high=7)
    metrics=analyze(params,examples,ticks)
    report={"protocol":"frozen-cellular-routing-lesion-v1",
            "checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "test_seed_root":seed, "episodes":episodes, "grid":size,
            "metrics":metrics,
            "limitations":"The destination query and a sender pulse are provided. "
                          "High source attention is not proof of object binding. "
                          "This is a frozen lesion test with no retraining; "
                          "it does not control for training compute or show "
                          "that routing aids visual prediction."}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",default="runs/cellular-routing-probe/metrics.json")
    p.add_argument("--seed",type=int,default=87000)
    p.add_argument("--episodes",type=int,default=32)
    p.add_argument("--size",type=int,default=10)
    args=p.parse_args()
    print(json.dumps(run(args.checkpoint,args.out,args.seed,args.episodes,args.size)
                     ["metrics"],indent=2))


if __name__=="__main__":
    main()
