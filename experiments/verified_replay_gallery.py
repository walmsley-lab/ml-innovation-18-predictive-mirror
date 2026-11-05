"""Offline recorded-video forecast gallery with exact future pairing and hashes.

This is not live inference, not a simulated timestamped issuance, and does not
claim a nominal frame horizon equals one elapsed second.
"""
import argparse
import hashlib
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from experiments.rgb_nca import read_recording
from experiments.rgb_transfer_eval import replay_windows
from src import predictive_rgb_nca as nca


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel_arrays(last,prediction,actual):
    if (last.shape!=prediction.shape or prediction.shape!=actual.shape or
            last.ndim!=3 or last.shape[-1]!=3):
        raise ValueError("Expected matched RGB images")
    return [last,prediction,actual,np.abs(prediction-actual)]


def save_panel(images,path,scale=8):
    from PIL import Image, ImageDraw
    captions=("Last observed","Predicted","Actual future","Absolute error")
    h,w,_=images[0].shape
    canvas=Image.new("RGB",(4*w*scale,(h+7)*scale),(20,30,45))
    painter=ImageDraw.Draw(canvas)
    for i,(name,img) in enumerate(zip(captions,images)):
        tile=Image.fromarray(np.asarray(np.clip(img,0,1)*255,np.uint8),"RGB")
        tile=tile.resize((w*scale,h*scale),Image.Resampling.NEAREST)
        canvas.paste(tile,(i*w*scale,7*scale))
        painter.text((i*w*scale+4,scale),name,fill=(245,248,250))
    canvas.save(path)


def run(checkpoint,recording,out="runs/verified-replay-gallery",context=8,
        horizon=10,max_windows=3,excluded_training_hashes=()):
    if context<2 or horizon<1 or max_windows<1:
        raise ValueError("Invalid forecast window")
    filehash=sha(recording)
    if filehash in set(excluded_training_hashes):
        raise ValueError("Evaluation recording used for training")
    frames,stamps=read_recording(recording)
    clips,anchors=replay_windows(frames,context,horizon,stride=5,cap=max_windows)
    with np.load(checkpoint,allow_pickle=False) as archive:
        weights={k:jnp.asarray(archive[k]) for k in archive.files}
    forecast=np.asarray(nca.rollout(
        weights,jnp.asarray(clips[:,:context]),horizon))
    target=clips[:,context:context+horizon]
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(len(clips)):
        image=folder/f"forecast-{i:02d}.png"
        save_panel(panel_arrays(clips[i,context-1],
                                forecast[i,-1],target[i,-1]),image)
        actual_index=int(anchors[i]+horizon)
        elapsed=None if stamps is None else float(
            (stamps[actual_index]-stamps[anchors[i]])/1e6)
        rows.append({"anchor_frame":int(anchors[i]),
                     "future_frame":actual_index,
                     "observed_elapsed_ms":elapsed,
                     "final_rgb_mse":float(np.mean(
                         (forecast[i,-1]-target[i,-1])**2)),
                     "gallery_file":image.name})
    report={"experiment":"offline_verified_camera_replay",
            "not_live":True,
            "checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),
            "recording":str(recording),"recording_sha256":filehash,
            "context":context,"horizon_frames":horizon,"pairs":rows,
            "warning":"Offline held-out replay, not proof a live system met a wall-clock deadline."}
    (folder/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--recording",required=True)
    p.add_argument("--out",default="runs/verified-replay-gallery")
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--max-windows",type=int,default=3)
    a=p.parse_args()
    print(json.dumps(run(a.checkpoint,a.recording,a.out,
                         a.context,a.horizon,a.max_windows),indent=2))


if __name__=="__main__":
    main()
