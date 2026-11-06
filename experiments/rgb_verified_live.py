"""Show actual t+10 against the *forecast issued ten frames earlier*."""
import argparse
import json
import time
from collections import deque
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from experiments.rgb_nca import preprocess_color
from src import rgb_checkpoint_predictor as predictor, rgb_delayed_scoring as delayed


def panel(pair, cv2, size=224):
    """Side-by-side observed anchor, issued forecast, realized future and error."""
    cols=("Observed at issue","Forecast +10","Actual at target","Abs error")
    images=(pair["observed"],pair["prediction"],pair["actual"],
            np.abs(pair["prediction"]-pair["actual"]))
    canvas=np.full((size+55,4*size,3),25,np.uint8)
    for j,(label,value) in enumerate(zip(cols,images)):
        image=(np.clip(value,0,1)*255).astype(np.uint8)
        bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        bgr=cv2.resize(bgr,(size,size),interpolation=cv2.INTER_NEAREST)
        canvas[50:50+size,j*size:(j+1)*size]=bgr
        cv2.putText(canvas,label,(j*size+4,25),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(200,230,240),1)
    return canvas


def run(parameters, output="runs/rgb-live-verified/metrics.json", camera=0,
        fps=10, size=32, context=8, lead=10, ticks=2,
        max_frames=0, no_display=False, model="nca"):
    if fps<=0 or context<2 or lead<1 or ticks<1:
        raise ValueError("Invalid camera settings")
    import cv2
    with np.load(parameters,allow_pickle=False) as archive:
        params={key:jnp.asarray(archive[key]) for key in archive.files}
    predictor.validate(params, model)
    predict=jax.jit(lambda x:predictor.rollout(params,x,lead,ticks,model))
    cap=cv2.VideoCapture(camera,cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB webcam {camera}")
    history=deque(maxlen=context)
    pending,paired=[],[]
    index=-1
    deadline=time.monotonic()
    inference_ms=[]
    try:
        while True:
            if deadline>time.monotonic():
                time.sleep(deadline-time.monotonic())
            ok,raw=cap.read()
            if not ok:
                break
            now=time.monotonic_ns()
            index+=1
            current=preprocess_color(raw,size,cv2)
            pending,matches=delayed.resolve(pending,index,now,current,lead)
            paired.extend(matches)
            if matches and not no_display:
                cv2.imshow("Verified RGB forecast vs actual",panel(matches[-1],cv2))
            history.append(current)
            if len(history)==context:
                before=time.perf_counter()
                forecast=np.asarray(predict(jnp.asarray(np.stack(history)[None])))[0]
                inference_ms.append((time.perf_counter()-before)*1000)
                # Timestamp after JAX materializes the forecast, not the
                # camera read that happened before inference.
                committed_ns = time.monotonic_ns()
                if committed_ns < now:
                    raise RuntimeError("Monotonic clock regressed")
                delayed.issue(pending,index,committed_ns,current,forecast)
            if not no_display and cv2.waitKey(1)&0xff in (27,ord("q")):
                break
            if max_frames and index+1>=max_frames:
                break
            deadline=max(deadline+1/fps,time.monotonic())
    finally:
        cap.release()
        if not no_display:
            cv2.destroyAllWindows()
    report=delayed.metrics(paired)
    report.update({"model":model,"lead_frames":lead,"frames_captured":index+1,
                   "median_inference_ms":float(np.median(inference_ms))
                                         if inference_ms else None,
                   "caveat":"Only paired predictions issued before target observation "
                            "are scored. Lead is camera-frame based, NOT guaranteed "
                            "one wall-clock second; inspect observed horizon milliseconds."})
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters",required=True)
    p.add_argument("--model", choices=predictor.MODELS, default="nca")
    p.add_argument("--out",default="runs/rgb-live-verified/metrics.json")
    p.add_argument("--camera",type=int,default=0)
    p.add_argument("--fps",type=float,default=10)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--lead",type=int,default=10)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--no-display",action="store_true")
    a=p.parse_args()
    print(json.dumps(run(a.parameters,a.out,a.camera,a.fps,a.size,
                         a.context,a.lead,a.ticks,a.max_frames,
                         a.no_display,a.model),indent=2))


if __name__=="__main__":
    main()
