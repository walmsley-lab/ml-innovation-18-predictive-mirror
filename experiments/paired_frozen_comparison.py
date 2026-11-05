"""One frozen, equal-input paired-history evaluation for saved video forecasts.

No training, model architecture selection, or checkpoint lookup happens here.
The caller must generate all predictions from past RGB only and declare any
structural prior before using a comparison to claim model advantage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src import paired_occlusion_histories as world
from src import reappearance_scoring as events


def artifact_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generated(seeds,context=8,horizon=10,size=32):
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Need unique independent scene-pair IDs")
    if context<8 or horizon<6 or size<32:
        raise ValueError("Paired world requires context>=8,horizon>=6,size>=32")
    records=[r for seed in seeds for r in world.pair(int(seed),context,horizon,size)]
    frames=np.stack([record["frames"] for record in records])
    return records,frames[:,:context],frames[:,context:context+horizon]


def model_forecast(path,shape,layout="bthwc"):
    with np.load(path,allow_pickle=False) as d:
        if set(d.files)!={"predictions"}:
            raise ValueError("Require exactly one 'predictions' array")
        x=np.asarray(d["predictions"],np.float32)
    if layout=="btchw":
        if x.ndim!=5:
            raise ValueError("Invalid channel-first forecast")
        x=x.transpose(0,1,3,4,2)
    elif layout!="bthwc":
        raise ValueError("Unknown layout")
    if x.shape!=shape or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError("Prediction shape/range mismatch")
    return x


def summarize(predicted,truth,records,context):
    if predicted.shape!=truth.shape:
        raise ValueError("Forecast and future truth differ in shape")
    ev=events.score(predicted,records,context)
    hit=[]
    for row in ev["results"]:
        if len(row["events"])!=1:
            raise ValueError("Expected one return event per episode")
        x=row["events"][0]
        hit.append(int(x["reappearance_centroid_error_px"] is not None
                       and x["reappearance_centroid_error_px"]<=3.
                       and x["timing_error_frames"] is not None
                       and abs(x["timing_error_frames"])<=1
                       and x["premature_visible_frames"]==0))
    if len(hit)%2:
        raise ValueError("Paired trials must be even")
    mse=np.square(predicted-truth).mean(axis=(0,2,3,4))
    return {"success_rate":float(np.mean(hit)),
            "scene_pair_success":np.asarray(hit).reshape(-1,2).mean(axis=1).tolist(),
            "mse_by_lead":mse.tolist(),
            "rgb_mse":float(mse.mean()),
            "future_difference_error":float(np.mean(
                ((predicted[::2]-predicted[1::2])-
                 (truth[::2]-truth[1::2]))**2)),
            "events":ev}


def run(out="runs/paired-frozen/metrics.json",seeds=(71000,71001,71002),
        forecasts=(),context=8,horizon=10):
    records,prefix,truth=generated(seeds,context,horizon)
    conditions={
        "persistence":{"tier":"rgb_only","forecast":
           np.repeat(prefix[:,-1:],horizon,axis=1),"artifact":None},
        "known_prior_linear":{"tier":"declared_known_prior","forecast":
           np.stack([world.linear_history_forecast(x,horizon) for x in prefix]),
           "artifact":None}
    }
    for spec in forecasts:
        # model_name:tier:layout:path.npz (path can contain ':' after third)
        name,tier,layout,path=spec.split(":",3)
        if name in conditions or tier not in ("rgb_only","declared_known_prior") or not name:
            raise ValueError("Model/tier invalid or duplicated")
        if layout not in ("bthwc","btchw"):
            raise ValueError("Bad prediction layout")
        conditions[name]={"tier":tier,
                          "forecast":model_forecast(path,truth.shape,layout),
                          "artifact":{"path":path,"sha256":artifact_sha(path)}}
    results={}
    for name,item in conditions.items():
        scored=summarize(item["forecast"],truth,records,context)
        results[name]={"input_tier":item["tier"],"artifact":item["artifact"],
                       **scored}
    comparable=[(name,value["success_rate"]) for name,value in results.items()
                if value["input_tier"]=="rgb_only"]
    report={"experiment":"paired_frozen_scoreboard_v1",
            "scene_pair_ids":list(map(int,seeds)),"context":context,
            "horizon":horizon,"input":"RGB history only for rgb_only tier",
            "results":results,
            "fair_ranking_by_event_success":sorted(comparable,key=lambda v:-v[1]),
            "missing_reference_note":"No learned reference was trained by this scorer; "
                                     "only models provided as frozen predictions are ranked"}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/paired-frozen/metrics.json")
    p.add_argument("--seeds",type=int,nargs="+",default=[71000,71001,71002])
    p.add_argument("--forecast",action="append",default=[],
       help="name:rgb_only|declared_known_prior:bthwc|btchw:path.npz")
    a=p.parse_args()
    r=run(a.out,a.seeds,a.forecast)
    print(json.dumps(r["fair_ranking_by_event_success"],indent=2))


if __name__=="__main__":
    main()
