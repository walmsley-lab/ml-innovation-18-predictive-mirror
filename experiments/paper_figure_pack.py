"""Make publication figures ONLY from real saved, matched forecast arrays.

No synthetic examples are silently generated here; missing data/provenance
fail closed. The training runner and checkpoint gallery must run first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(arrays):
    required=("observed","nca","persistence","velocity","truth")
    if any(x not in arrays for x in required):
        raise ValueError("Figure input needs observed, nca, persistence, velocity, truth")
    future=np.asarray(arrays["truth"])
    if future.ndim!=5 or future.shape[-1]!=3 or future.shape[1]<1:
        raise ValueError("Expected future RGB [N,Horizon,H,W,3]")
    for key in ("nca","persistence","velocity"):
        value=np.asarray(arrays[key])
        if value.shape!=future.shape or not np.isfinite(value).all():
            raise ValueError("Incompatible or nonfinite model forecast: "+key)
    observed=np.asarray(arrays["observed"])
    if observed.shape!=(future.shape[0],*future.shape[2:]) or not np.isfinite(observed).all():
        raise ValueError("Invalid last observed RGB")
    if any(np.min(np.asarray(arrays[k]))<0 or np.max(np.asarray(arrays[k]))>1
           for k in required):
        raise ValueError("RGB intensities must be normalized [0,1]")
    return future


def metrics(arrays,leads=(1,3,5,10)):
    truth=validate(arrays)
    output={}
    for lead in leads:
        if lead<1 or lead>truth.shape[1]:
            continue
        key=str(lead)
        target=truth[:,lead-1]
        output[key]={model:float(np.mean(
            (np.asarray(arrays[model])[:,lead-1]-target)**2))
            for model in ("nca","persistence","velocity")}
    if not output:
        raise ValueError("No requested lead inside prediction horizon")
    return output


def image_strip(arrays,episode,lead,scale=5):
    from PIL import Image,ImageDraw
    truth=validate(arrays)
    if episode<0 or episode>=truth.shape[0] or lead<1 or lead>truth.shape[1]:
        raise ValueError("Invalid episode/lead")
    last=arrays["observed"][episode]
    predicted=arrays["nca"][episode,lead-1]
    controls=[arrays[k][episode,lead-1] for k in ("persistence","velocity")]
    target=truth[episode,lead-1]
    error=np.abs(predicted-target)
    panels=[last,predicted,*controls,target,error]
    names=["Last observed","NCA","Persistence","RGB velocity",
           "Future truth","Abs error"]
    height,width,_=target.shape
    out=Image.new("RGB",(6*width*scale,(height+8)*scale),(22,34,48))
    draw=ImageDraw.Draw(out)
    for i,(name,frame) in enumerate(zip(names,panels)):
        tile=Image.fromarray(np.asarray(np.clip(frame,0,1)*255,np.uint8),"RGB")
        tile=tile.resize((width*scale,height*scale),Image.Resampling.NEAREST)
        out.paste(tile,(i*width*scale,8*scale))
        draw.text((i*width*scale+3,scale),name,fill=(238,246,250))
    return out


def run(data_path,provenance_path,out="runs/paper-figures",
        leads=(1,3,5,10),episode=0):
    provenance=json.loads(Path(provenance_path).read_text())
    fields=("checkpoint_sha256","code_revision","test_scene_seeds",
            "generator","protocol")
    if any(not provenance.get(x) for x in fields):
        raise ValueError("Incomplete measured-figure provenance")
    with np.load(data_path,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    score=metrics(arrays,leads)
    folder=Path(out)
    folder.mkdir(parents=True,exist_ok=True)
    figures=[]
    for lead in score:
        name=f"rgb-comparison-lead-{lead}.png"
        image_strip(arrays,episode,int(lead)).save(folder/name)
        figures.append(name)
    report={"status":"derived from supplied saved arrays; no training performed",
            "input_sha256":sha(data_path),"provenance_sha256":sha(provenance_path),
            "provenance":provenance,"episode_index":episode,
            "mse_by_lead":score,"figures":figures,
            "warning":"Validity still depends on source checkpoint and inference code."}
    (folder/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True,help="NPZ with observed,nca,persistence,velocity,truth")
    p.add_argument("--provenance",required=True,help="JSON with checkpoint/run/git provenance")
    p.add_argument("--out",default="runs/paper-figures")
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--leads",type=int,nargs="+",default=[1,3,5,10])
    a=p.parse_args()
    print(json.dumps(run(a.data,a.provenance,a.out,a.leads,a.episode),indent=2))


if __name__=="__main__":
    main()
