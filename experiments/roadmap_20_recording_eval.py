"""Roadmap 20: independent timestamped webcam-recording evaluation.

Strict causal OFFLINE replay. Model sees only frames through issue index;
future observations are scoring labels. Offline processing runtime must NOT
be reported as live camera issuance latency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments import frozen_visual_bridge as bridge
from src import rgb_benchmark as benchmark


def session(path,context=4,lead=3):
    p=Path(path)
    with np.load(p,allow_pickle=False) as data:
        if "frames" not in data.files or "timestamps_ns" not in data.files:
            raise ValueError("Require acquired RGB frames and monotonic timestamps_ns")
        frames=np.asarray(data["frames"],np.float32)
        stamps=np.asarray(data["timestamps_ns"],np.int64)
    if (frames.ndim!=4 or frames.shape[-1]!=3 or
        len(frames)!=len(stamps) or len(frames)<context+lead+1 or
        not np.isfinite(frames).all() or np.min(frames)<0 or np.max(frames)>1 or
        np.any(np.diff(stamps)<=0)):
        raise ValueError("Invalid capture shape/range/timestamps")
    return {"frames":frames,"timestamps_ns":stamps,
            "source_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
            "source_path":str(p)}


def evaluate(params,name,record,context=4,lead=3,ticks=1):
    frames=record["frames"]
    stamps=record["timestamps_ns"]
    errors={"model":[],"persistence":[],"constant_velocity":[]}
    horizons=[]
    for issue_index in range(context-1,len(frames)-lead):
        prefix=frames[issue_index-context+1:issue_index+1][None]
        forecast=bridge.predict(name,params,prefix,lead,ticks)[0,lead-1]
        controls=benchmark.controls(prefix,lead)
        actual=frames[issue_index+lead]
        errors["model"].append(float(np.square(forecast-actual).mean()))
        for model,series in controls.items():
            errors[model].append(float(np.square(series[0,lead-1]-actual).mean()))
        horizons.append(float((stamps[issue_index+lead]-stamps[issue_index])/1e6))
    return {"scores_mse":{name:float(np.mean(values)) for name,values in errors.items()},
            "paired_targets":len(horizons),
            "actual_horizon_ms_median":float(np.median(horizons)),
            "actual_horizon_ms_p05":float(np.percentile(horizons,5)),
            "actual_horizon_ms_p95":float(np.percentile(horizons,95)),
            "time_contract":"post-recording replay, NOT measured live issuance"}


def run(out,calibration,test,model,checkpoint=None,smoke=False,
        context=4,lead=3,width=8,ticks=1):
    a=session(calibration,context,lead)
    b=session(test,context,lead)
    if a["source_sha256"]==b["source_sha256"]:
        raise ValueError("Calibration and test recordings must not be identical")
    params,meta=bridge.checkpoint(model,checkpoint,smoke,width)
    # Keep calibration and test separate even if calibration not used by
    # this fixed model. Never use test outcomes to select a checkpoint.
    assessment=evaluate(params,model,b,context,lead,ticks)
    report={"roadmap_id":20,"model":model,"checkpoint":meta,
            "calibration_source_sha256":a["source_sha256"],
            "test_source_sha256":b["source_sha256"],
            "test":assessment,"context":context,"lead_frames":lead,
            "limits":"Distinct file hashes do not establish separate capture "
            "sessions or an unseen test cohort. Model computation happens "
            "offline; no live latency assertion. Input action is unknown. "
            "Require trained checkpoints for scientific evidence."}
    if out:
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration",required=True)
    p.add_argument("--test",required=True)
    p.add_argument("--model",choices=bridge.MODELS,default="nca")
    p.add_argument("--checkpoint")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--context",type=int,default=4)
    p.add_argument("--lead",type=int,default=3)
    p.add_argument("--width",type=int,default=8)
    p.add_argument("--out",default="runs/roadmap-20/metrics.json")
    a=p.parse_args()
    print(json.dumps(run(a.out,a.calibration,a.test,a.model,a.checkpoint,
                          a.smoke,a.context,a.lead,a.width)["test"],indent=2))


if __name__=="__main__":
    main()
