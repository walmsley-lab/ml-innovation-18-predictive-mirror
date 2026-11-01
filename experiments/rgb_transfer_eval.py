"""Evaluate frozen RGB NCA forecasts on independent real camera recordings.

No webcam recording is used for training here. Explicitly exclude the
training file hash and report actual camera timestamp intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from experiments.rgb_nca import read_recording
from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_windows(frames, context=8, horizon=10, stride=5, cap=48):
    if context < 2 or horizon < 1 or stride < 1 or cap < 1:
        raise ValueError("Invalid window settings")
    # Context and targets must both remain inside final 20% holdout.
    first=int(.8*len(frames))+context-1
    last=len(frames)-horizon-1
    if first>last:
        raise ValueError("Recording too short for final-20% holdout")
    anchors=list(range(first,last+1,stride))[:cap]
    clips=[frames[i-context+1:i+horizon+1] for i in anchors]
    return np.stack(clips),anchors


def evaluate(checkpoint, recordings, output="runs/rgb-transfer/metrics.json",
             context=8, horizon=10, ticks=2, stride=5, cap=48,
             excluded_training_hashes=()):
    if not recordings:
        raise ValueError("Provide at least one independent RGB recording")
    with np.load(checkpoint,allow_pickle=False) as archive:
        params={k:jnp.asarray(archive[k]) for k in archive.files}
    excluded=set(excluded_training_hashes)
    rows=[]
    for path in recordings:
        digest=sha(path)
        if digest in excluded:
            raise ValueError(f"Training recording reused for evaluation: {path}")
        frames,stamps=read_recording(path)
        clips,anchors=replay_windows(frames,context,horizon,stride,cap)
        # Predict in bounded batches to avoid large CPU allocations.
        predictions=[]
        for block in np.array_split(clips,max(1,int(np.ceil(len(clips)/4)))):
            predictions.append(np.asarray(nca.rollout(
                params,jnp.asarray(block[:,:context]),horizon,ticks)))
        predicted=np.concatenate(predictions)
        leads=tuple(h for h in benchmark.LEADS if h<=horizon)
        metrics=benchmark.compare(predicted,clips,context,leads)
        observed_ms=float(np.median(np.diff(stamps).astype(np.float64))/1e6) if (
            stamps is not None and len(stamps)>1) else None
        rows.append({"recording":str(path),"sha256":digest,
                     "frames":len(frames),"holdout_windows":len(anchors),
                     "median_capture_interval_ms":observed_ms,
                     "nominal_horizon_ms":horizon*observed_ms if observed_ms else None,
                     "scores":metrics})
    report={"experiment":"independent_rgb_webcam_transfer","checkpoint":str(checkpoint),
            "context":context,"horizon_frames":horizon,
            "records":rows,"caveat":"Checkpoints must not have seen any evaluation "
                "recording or overlapping scene; explicit hashes guard only known "
                "training files. Inference predicts future frames without viewing them. "
                "A 10-frame horizon is 1 second only with ~100 ms camera intervals."}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--recordings",nargs="+",required=True)
    p.add_argument("--exclude-training",nargs="*",default=[])
    p.add_argument("--out",default="runs/rgb-transfer/metrics.json")
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--stride",type=int,default=5)
    p.add_argument("--max-windows",type=int,default=48)
    a=p.parse_args()
    blocked=[sha(path) for path in a.exclude_training]
    report=evaluate(a.checkpoint,a.recordings,a.out,a.context,a.horizon,a.ticks,
                    a.stride,a.max_windows,blocked)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
