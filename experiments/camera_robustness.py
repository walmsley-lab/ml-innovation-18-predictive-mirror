"""Synthetic characterization and optional real-camera metadata capture."""
import argparse
import json
from pathlib import Path
import numpy as np
from src import camera_robustness as sensor, rgb_motion_world as world


def run(out="runs/camera-robustness/metrics.json",seeds=(11,23,41),
        context=8,horizon=5,size=32,drop_rates=(.01,.05,.10)):
    if not seeds or context<2 or horizon<1:
        raise ValueError("Invalid camera study")
    modes=[("none",0),("exposure",0),("white_balance",0),
           ("rolling_shutter",0)]
    modes.extend((mode,rate) for mode in ("isolated_drop","burst_drop")
                 for rate in drop_rates)
    results=[]
    for seed in seeds:
        truth=world.episode(seed,steps=context+horizon,size=size)["frames"]
        for mode,rate in modes:
            sample=sensor.corrupt(truth,mode,rate,seed)
            observations=sample["frames"]
            # Predict from the corrupted observed history only. Future
            # sensor truth is held back until evaluation.
            predicted=np.repeat(observations[context-1:context],horizon,axis=0)
            per_frame=np.square(predicted-observations[context:]).mean(
                axis=(1,2,3))
            capture=sample["capture"][context:]
            results.append({"seed":int(seed),"mode":mode,"drop_rate":rate,
                "missing_observed_frames":int(sample["missing"][:context].sum()),
                "target_mse":float(per_frame.mean()),
                "strata":sensor.strata(per_frame,capture,
                        sample["missing"][context:]),
                "nominal_lead_ms":horizon*100,
                "capture_period_ms":100})
    report={"protocol":"camera-robustness-v1","results":results,
            "limits":"Synthetic disturbances and persistence baseline only. "
            "Capture timestamps are simulated at 100 ms; they are not actual "
            "camera hardware measurements. Exposure prediction may improve "
            "sensor RGB MSE without improving physical motion. "
            "Real rolling shutter, auto settings and dropped captures must "
            "be measured on actual hardware separately."}
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


def capture_metadata(index=0,frames=60,lock=False):
    """Optional OpenCV backend. Attempts locks and reports whether accepted."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Real camera capture needs opencv-python") from error
    import time
    handle=cv2.VideoCapture(index)
    if not handle.isOpened():
        raise RuntimeError("Camera could not be opened")
    caps={"exposure":cv2.CAP_PROP_EXPOSURE,"gain":cv2.CAP_PROP_GAIN,
          "white_balance":getattr(cv2,"CAP_PROP_WB_TEMPERATURE",None)}
    accepted={}
    readings=[]
    try:
        if lock:
            requested={"auto_exposure":(cv2.CAP_PROP_AUTO_EXPOSURE,1.),
                       "auto_white_balance":(getattr(cv2,"CAP_PROP_AUTO_WB",None),0.)}
            for key,(setting,value) in requested.items():
                accepted[key]=bool(handle.set(setting,value)) if setting is not None else None
        for _ in range(frames):
            ok,image=handle.read()
            stamp=time.monotonic()
            if not ok:continue
            entry={"timestamp_s":stamp}
            for key,prop in caps.items():
                value=handle.get(prop) if prop is not None else None
                # OpenCV 0 is ambiguous: do not pretend unsupported properties are known.
                entry[key]=float(value) if value is not None and np.isfinite(value) else None
            readings.append(entry)
    finally:
        handle.release()
    return {"records":readings,"requested_locks":accepted,
            "warning":"Driver values and lock-success reports must be validated "
                      "per device; no 0-valued property proves physical exposure."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/camera-robustness/metrics.json")
    p.add_argument("--real-camera",action="store_true")
    p.add_argument("--camera-index",type=int,default=0)
    p.add_argument("--frames",type=int,default=60)
    p.add_argument("--lock",action="store_true")
    a=p.parse_args()
    results=capture_metadata(a.camera_index,a.frames,a.lock) if a.real_camera else run(a.out)
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
