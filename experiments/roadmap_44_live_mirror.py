"""44: live, selected RGB forecaster with timestamped actual-frame verification.

Predictions are materialized and timestamped BEFORE future camera frames
arrive. Results include persistence and image-velocity controls. No model
selection is performed on the live test stream.
"""
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
import time
import numpy as np
from experiments import frozen_visual_bridge as bridge
from experiments.rgb_nca import preprocess_color
from src import rgb_delayed_scoring as delayed, rgb_benchmark


def selected(selection_path,checkpoint):
    report=json.loads(Path(selection_path).read_text())
    chosen=report.get("selected")
    if not chosen or not chosen.get("checkpoint_sha256"):
        raise ValueError("Require a validated model-selection manifest")
    if not checkpoint:
        raise ValueError("Selected trained checkpoint path required")
    digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if chosen["checkpoint_sha256"]!=digest:
        raise ValueError("Selected checkpoint digest mismatch")
    if chosen["name"] not in bridge.MODELS:
        raise ValueError("Unsupported selected visual model")
    return chosen


def aligned_metrics(channels):
    rows={}
    for name,pairs in channels.items():
        usable=[pair for pair in pairs if pair["elapsed_ms"]>0]
        score=delayed.metrics(usable)
        score["late_predictions"]=len(pairs)-len(usable)
        rows[name]=score
    return rows


def replay_stream(frames,stamps,model,params,context=4,lead=2,ticks=1):
    """Deterministic functional harness, NOT an actual webcam latency test."""
    if len(frames)!=len(stamps) or np.any(np.diff(stamps)<=0):
        raise ValueError("Need aligned monotonic frame timestamps")
    pending={name:[] for name in ("selected","persistence","constant_velocity")}
    scored={name:[] for name in pending}
    for index,frame in enumerate(frames):
        for name in pending:
            pending[name],pairs=delayed.resolve(
                pending[name],index,int(stamps[index]),frame,lead)
            scored[name].extend(pairs)
        if index+1<context:
            continue
        observed=frames[index-context+1:index+1][None]
        selected_forecast=bridge.predict(model,params,observed,lead,ticks)[0]
        controls=rgb_benchmark.controls(observed,lead)
        for name,forecast in (("selected",selected_forecast),
                 ("persistence",controls["persistence"][0]),
                 ("constant_velocity",controls["constant_velocity"][0])):
            delayed.issue(pending[name],index,int(stamps[index]),
                          frame,forecast)
    return aligned_metrics(scored)


def live(selection_path,checkpoint,out="runs/roadmap-44/live.json",
         camera=0,fps=10,size=32,context=8,lead=10,width=8,ticks=2,
         max_frames=0,no_display=False):
    if fps<=0 or context<2 or lead<1 or ticks<1:
        raise ValueError("Invalid live sampling contract")
    import cv2
    chosen=selected(selection_path,checkpoint)
    params,_=bridge.checkpoint(chosen["name"],checkpoint,False,width)
    cap=cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to access camera")
    history=deque(maxlen=context)
    pending={key:[] for key in ("selected","persistence","constant_velocity")}
    scored={key:[] for key in pending}
    inference_ms=[]
    frames_seen=0
    next_capture=time.monotonic()
    try:
        while max_frames<=0 or frames_seen<max_frames:
            now=time.monotonic()
            if now<next_capture:
                time.sleep(next_capture-now)
            ok,raw=cap.read()
            if not ok:
                break
            acquired=time.monotonic_ns()
            frame=preprocess_color(raw,size,cv2)
            for name in pending:
                pending[name],pairs=delayed.resolve(
                    pending[name],frames_seen,acquired,frame,lead)
                scored[name].extend(pairs)
            history.append(frame)
            if len(history)==context:
                h=np.stack(history)[None]
                started=time.perf_counter()
                forecast=bridge.predict(chosen["name"],params,h,lead,ticks)[0]
                inference_ms.append((time.perf_counter()-started)*1000)
                controls=rgb_benchmark.controls(h,lead)
                issued=time.monotonic_ns()
                for name,value in (("selected",forecast),
                        ("persistence",controls["persistence"][0]),
                        ("constant_velocity",controls["constant_velocity"][0])):
                    delayed.issue(pending[name],frames_seen,issued,frame,value)
            frames_seen+=1
            next_capture=max(next_capture+1/fps,time.monotonic())
            if not no_display and scored["selected"]:
                # Draw the requested lead, not a hardcoded "+10" label.
                pair=scored["selected"][-1]
                title=f"Forecast +{lead} captured frames"
                from experiments.rgb_verified_live import panel as verified_panel
                canvas=verified_panel(pair,cv2)
                # Replace only the forecast title band; the verifier owns
                # the actual/issued/error panels and their timestamp pairing.
                cv2.rectangle(canvas,(224,0),(448,42),(25,25,25),-1)
                cv2.putText(canvas,title,(228,25),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,(200,230,240),1)
                cv2.imshow("Selected mirror: observed / issued / actual",canvas)
                if cv2.waitKey(1)&0xff in (27,ord("q")):
                    break
    finally:
        cap.release()
        if not no_display:
            cv2.destroyAllWindows()
    report={"roadmap_id":44,"selected":chosen,
            "models":aligned_metrics(scored),
            "frames_captured":frames_seen,"lead_frames":lead,
            "median_inference_ms":float(np.median(inference_ms)) if inference_ms else None,
            "limits":"Lead specified in acquired frames, not guaranteed wall-clock "
            "seconds. Timestamp pairs with nonpositive elapsed time excluded. "
            "Unknown camera action; live session is test-only, never tuning."}
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--camera",type=int,default=0)
    p.add_argument("--fps",type=float,default=10)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--lead",type=int,default=10)
    p.add_argument("--width",type=int,default=8)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--no-display",action="store_true")
    p.add_argument("--out",default="runs/roadmap-44/live.json")
    a=p.parse_args()
    print(json.dumps(live(a.selection,a.checkpoint,a.out,a.camera,a.fps,a.size,
                          a.context,a.lead,a.width,2,a.max_frames,
                          a.no_display),indent=2))


if __name__=="__main__":
    main()
