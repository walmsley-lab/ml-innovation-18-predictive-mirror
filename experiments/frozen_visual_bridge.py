"""Frozen visual-model evaluation with episode-level conformal calibration.

No training occurs here. Models get only RGB history. Synthetic simulator state
is never passed to predictors; camera recordings must be separately supplied.
Frozen checkpoints are REQUIRED for research comparisons; --smoke permits
untrained initialization but labels it explicitly as non-evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import jax
import jax.numpy as jnp
from src import (predictive_rgb_nca as nca, tiny_cnn_video as cnn,
                 rgb_motion_transport as transport, rgb_world_ladder as world,
                 camera_robustness as sensor, rgb_benchmark as bench)

MODELS = ("nca", "cnn", "transport")
DOMAINS = ("linear", "accelerating", "occlusion_long", "camera_pan", "combined")
CORRUPTIONS = ("none", "exposure", "white_balance", "isolated_drop")


def initialize(name, seed=11, width=8):
    if name == "nca":
        return nca.init(jax.random.PRNGKey(seed), width)
    if name == "cnn":
        return cnn.init(jax.random.PRNGKey(seed), width)
    if name == "transport":
        return transport.init(jax.random.PRNGKey(seed), width)
    raise ValueError("Unknown model family")


def checkpoint(name, path=None, smoke=False, width=8):
    reference = initialize(name, width=width)
    if path is None:
        if not smoke:
            raise ValueError("Frozen trained checkpoint required; --smoke for untrained diagnostics")
        return reference, {"checkpoint_sha256": None, "evidence": "untrained_smoke"}
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(reference):
            raise ValueError("Checkpoint parameter keys do not match this architecture")
        params = {}
        for key, shape in reference.items():
            value = np.asarray(saved[key])
            if value.shape != shape.shape or not np.isfinite(value).all():
                raise ValueError("Checkpoint shape or finite-value mismatch: " + key)
            params[key] = jnp.asarray(value)
    return params, {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "evidence": "checkpoint_loaded_training_not_independently_verified"}


def predict(name, params, history, horizon, ticks=2):
    x = jnp.asarray(history, jnp.float32)
    if name == "nca":
        return np.asarray(nca.rollout(params, x, horizon, ticks))
    if name == "cnn":
        return np.asarray(cnn.rollout(params, x, horizon))
    if name == "transport":
        return np.asarray(transport.rollout(params, x, horizon, "combined"))
    raise ValueError(name)


def worlds(first_seed, count, context, horizon, size, domain, corruption="none"):
    records = [world.episode(first_seed+i, context+horizon, size=size,
                             domain=domain) for i in range(count)]
    clean = np.stack([r["frames"] for r in records])
    observed = np.stack([sensor.corrupt(frames, corruption, .05, first_seed+i)["frames"]
                         for i, frames in enumerate(clean)])
    return observed, clean


def recorded(path, length):
    with np.load(path, allow_pickle=False) as data:
        if "frames" not in data.files:
            raise ValueError("Recording must supply frames")
        rgb = np.asarray(data["frames"], np.float32)
    if rgb.ndim != 5 or rgb.shape[-1] != 3 or rgb.shape[1] < length or (
            not np.isfinite(rgb).all()) or np.min(rgb)<0 or np.max(rgb)>1:
        raise ValueError("Need independent normalized [episodes,time,height,width,3]")
    return rgb[:, :length]


def radius(scores, alpha=.1):
    """Conformal quantile over independent EPISODE maxima, per lead.

    Intervals are simultaneous across image pixels/channels for each lead
    under exchangeable episodes; not guaranteed during camera domain shift.
    """
    a=np.asarray(scores,float)
    if a.ndim!=2 or not 0<alpha<1 or not np.isfinite(a).all():
        raise ValueError("Need [episodes,lead] nonconformity and alpha")
    rank=math.ceil((len(a)+1)*(1-alpha))
    if rank>len(a):
        raise ValueError("Insufficient independent calibration episodes")
    return np.sort(a,axis=0)[rank-1]


def calibrate(predicted, actual, alpha=.1):
    if predicted.shape!=actual.shape or predicted.ndim!=5:
        raise ValueError("Forecast/target mismatch")
    residual=np.max(np.abs(predicted-actual),axis=(2,3,4))
    return radius(residual,alpha)


def coverage(predicted, actual, radii):
    residual=np.max(np.abs(predicted-actual),axis=(2,3,4))
    if radii.shape!=(residual.shape[1],):
        raise ValueError("Calibration horizon mismatch")
    return {"simultaneous_pixel_coverage_by_lead":
                np.mean(residual<=radii[None,:],axis=0).tolist(),
            "mean_interval_width_by_lead":(2*radii).tolist()}


def evaluate(name, params, calibration, test, context, horizon, alpha=.1, ticks=2):
    calibrated=predict(name,params,calibration[:,:context],horizon,ticks)
    radii=calibrate(calibrated,calibration[:,context:context+horizon],alpha)
    forecast=predict(name,params,test[:,:context],horizon,ticks)
    baseline=bench.controls(test[:,:context],horizon)
    leads=tuple(sorted({1,horizon}))
    results={name:bench.score(forecast,test,context,leads)}
    results.update({k:bench.score(v,test,context,leads) for k,v in baseline.items()})
    return {"metrics":results,"conformal_radius_by_lead":radii.tolist(),
            "coverage":coverage(forecast,test[:,context:context+horizon],radii)}


def run(out="runs/visual-bridge/frozen.json", checkpoints=None, smoke=False,
        calibration_count=24, test_count=12, context=8, horizon=5, size=24,
        width=8, ticks=2, alpha=.1, calibration_recording=None,
        test_recording=None):
    if min(calibration_count,test_count,context,horizon,size,width,ticks)<1 or context<2:
        raise ValueError("Bad evaluation budget")
    if bool(calibration_recording)!=bool(test_recording):
        raise ValueError("Supply both independent calibration and test recordings")
    checkpoints=checkpoints or {}
    models={name:checkpoint(name,checkpoints.get(name),smoke,width)
            for name in MODELS}
    recording_mode=bool(test_recording)
    if recording_mode:
        a,b=Path(calibration_recording),Path(test_recording)
        if hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest():
            raise ValueError("Calibration and test recordings must differ")
        cal=recorded(a,context+horizon)
        test=recorded(b,context+horizon)
        cases=[("independent_recordings",cal,test)]
    else:
        cases=[]
        for domain in DOMAINS:
            cal,_=worlds(61000,calibration_count,context,horizon,size,domain)
            for corruption in CORRUPTIONS:
                obs,clean=worlds(91000,test_count,context,horizon,size,
                                 domain,corruption)
                # Score sensor-observed future for none; corruption during
                # inference is a stress test scored against clean future.
                target=obs if corruption=="none" else clean
                test=np.concatenate((obs[:,:context],target[:,context:]),axis=1)
                cases.append((domain+"/"+corruption,cal,test))
    report={"protocol":"frozen-visual-bridge-v1","context":context,"horizon":horizon,
            "alpha":alpha,"checkpoints":{k:meta for k,(_,meta) in models.items()},
            "cases":{},"recording_mode":recording_mode,
            "limits":"Checkpoints are supplied by caller. Smoke initializes untrained models. "
            "Synthetic corruption tests score clean future against corrupted past; "
            "future corruption is not forecast. Calibration from clean source episodes "
            "is not a shifted-domain coverage guarantee. For recordings, independence "
            "of acquisition sessions cannot be proven from unequal files alone."}
    for label,cal,test in cases:
        report["cases"][label]={name:evaluate(name,p,cal,test,context,horizon,alpha,ticks)
                                for name,(p,_) in models.items()}
    if out:
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/visual-bridge/frozen.json")
    p.add_argument("--nca-checkpoint")
    p.add_argument("--cnn-checkpoint")
    p.add_argument("--transport-checkpoint")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--calibration-recording")
    p.add_argument("--test-recording")
    p.add_argument("--count",type=int,default=12)
    a=p.parse_args()
    print(json.dumps(run(a.out,{"nca":a.nca_checkpoint,"cnn":a.cnn_checkpoint,
         "transport":a.transport_checkpoint},a.smoke,test_count=a.count,
         calibration_recording=a.calibration_recording,
         test_recording=a.test_recording)["checkpoints"],indent=2))


if __name__=="__main__":
    main()
