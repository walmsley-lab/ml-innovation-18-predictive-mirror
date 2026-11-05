"""Roadmap 30: causal reconciliation between pixel evidence and compact motion state.

A dense saliency representation and a 2D position/velocity state are updated
independently or coupled through spatial feedback. The same update budget
is used for feedback/no-feedback and conflicting-context controls.
This is an interpretable reference, not a trained NCA circuit.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def world(seed=11, steps=12, size=28):
    if size < 24 or steps < 5:
        raise ValueError("World too short or small")
    rng=np.random.default_rng(seed)
    frames=np.zeros((steps,size,size,3),np.float32)
    positions=np.zeros((steps,2),float)
    visible=np.ones(steps,bool)
    for t in range(steps):
        frame=np.full((size,size,3),.045,np.float32)
        x=int(4+1.5*t)
        y=int(size//2+seed%3-1)
        positions[t]=[x,y]
        visible[t]=t not in (5,6)
        if visible[t] and x<size-2:
            frame[y-1:y+2,x-1:x+2]=[.98,.06,.06]
        for _ in range(3):
            a,b=rng.integers(2,size-2,size=2)
            frame[b,a]=[.78,.04,.04]
        frames[t]=frame
    return frames,positions,visible


def saliency(rgb):
    image=np.asarray(rgb,float)
    return np.maximum(image[...,0]-np.maximum(image[...,1],image[...,2]),0.)


def decode(salience,prior=None,scale=4.):
    s=np.asarray(salience,float).copy()
    h,w=s.shape
    yy,xx=np.mgrid[:h,:w]
    if prior is not None:
        distance=(xx-prior[0])**2+(yy-prior[1])**2
        s*=np.exp(-distance/(2*scale*scale))
    total=float(s.sum())
    if total < 1e-7:
        return None
    return np.array([float((s*xx).sum()/total),
                     float((s*yy).sum()/total)])


def infer_series(frames,mode="feedback",other=None,ticks=2):
    if mode not in ("feedback","no_feedback","shuffled") or ticks<1:
        raise ValueError("Invalid feedback ablation")
    if mode=="shuffled" and (other is None or len(other)<len(frames)):
        raise ValueError("Need same-length independent distractor history")
    initial=decode(saliency(frames[0]))
    if initial is None:
        raise ValueError("First object must be observable")
    slot=initial.copy()
    velocity=np.zeros(2)
    outputs=[]
    for t,frame in enumerate(frames):
        stimulus=saliency(frame)
        prior=slot+velocity if t>0 else slot
        context=prior.copy()
        if mode=="shuffled":
            # Rival estimate comes from another *observed* image at same t.
            context=decode(saliency(other[t]))
            if context is None:
                context=prior.copy()
        pixel=None
        updated=prior.copy()
        for _ in range(ticks):
            pixel=decode(stimulus,context if mode!="no_feedback" else None)
            if pixel is None:
                pixel=prior.copy()
            updated=.55*prior+.45*pixel
            if mode=="feedback":
                context=updated
        velocity=.65*velocity+.35*(updated-slot)
        slot=updated
        outputs.append({"pixel":pixel.copy(),"slot":slot.copy(),
                        "agreement":float(np.linalg.norm(pixel-slot))})
    return outputs


def measures(output,truth,visible):
    pixels=np.stack([x["pixel"] for x in output])
    slots=np.stack([x["slot"] for x in output])
    return {"pixel_visible_mae_px":float(np.linalg.norm(pixels[visible]-truth[visible],axis=-1).mean()),
            "slot_visible_mae_px":float(np.linalg.norm(slots[visible]-truth[visible],axis=-1).mean()),
            "slot_hidden_mae_px":float(np.linalg.norm(slots[~visible]-truth[~visible],axis=-1).mean()),
            "cross_representation_disagreement_px":float(np.mean([x["agreement"] for x in output]))}


def run(out="runs/development-30/metrics.json",seeds=range(91000,91024),ticks=2):
    results={"feedback":[],"no_feedback":[],"shuffled":[]}
    for seed in seeds:
        frames,truth,visible=world(int(seed))
        other=world(int(seed)+10000)[0]
        for name in results:
            report=measures(infer_series(frames,name,other,ticks),truth,visible)
            results[name].append(report)
    scores={name:{key:float(np.mean([row[key] for row in rows]))
                  for key in rows[0]} for name,rows in results.items()}
    report={"roadmap_id":30,"test_episodes":len(seeds),"ticks":ticks,
            "scores":scores,
            "limits":"Heuristic iterative pixel-to-state and state-to-pixel feedback, "
                     "equal update counts. Same RGB, distinct representations, "
                     "no oracle input. Agreement without correct location is not "
                     "a reconciliation success. Neural feedback benefit unproven."}
    if out is not None:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/development-30/metrics.json")
    a=p.parse_args()
    print(json.dumps(run(a.out),indent=2))


if __name__=="__main__":
    main()
