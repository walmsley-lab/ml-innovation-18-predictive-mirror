"""Roadmap 39: particle NCA on observed-RGB centers versus privileged tracks.

The same particle model is trained on supplied state or image-extracted
coordinates. RGB pathway knows a fixed color palette, not hidden simulator
positions. Missing detections use causal last-seen holds; hidden truth is
evaluation-only. Never equate this with learned RGB object discovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import particle_nca as nca, rgb_motion_world as motion
from experiments import particle_nca_motion as task
from train import adamw

COLORS=motion.COLORS


def render(positions,size=24,occluded=None):
    """Render observed colored markers, no oracle identity in readout."""
    if positions.ndim!=3 or positions.shape[-1]!=2 or size<16:
        raise ValueError("Need [T,particles,2] positions")
    tmax,n,_=positions.shape
    if n>len(COLORS):
        raise ValueError("Synthetic palette supports up to four particles")
    if occluded is None:occluded=np.zeros((tmax,n),bool)
    yy,xx=np.mgrid[:size,:size]
    frames=np.full((tmax,size,size,3),.035,np.float32)
    for t in range(tmax):
        for j,(x,y) in enumerate(positions[t]):
            if occluded[t,j]:
                continue
            mask=(xx-x*(size-1))**2+(yy-y*(size-1))**2<=1.5**2
            frames[t,mask]=COLORS[j]
    return frames


def observe(frames,particles=2):
    """Palette-based RGB centroid detections, causal last-seen imputation."""
    tmax,h,w,_=frames.shape
    values=np.zeros((tmax,particles,2),np.float32)
    seen=np.zeros((tmax,particles),bool)
    yy,xx=np.mgrid[:h,:w]
    last=np.full((particles,2),.5,np.float32)
    for t in range(tmax):
        for j in range(particles):
            mask=np.max(np.abs(frames[t]-COLORS[j]),axis=-1)<.05
            if mask.any():
                last[j]=[np.mean(xx[mask])/(w-1),np.mean(yy[mask])/(h-1)]
                seen[t,j]=True
            values[t,j]=last[j]
    return values,seen


def pools(ids,context,horizon,size,particles,occlude=False):
    sequences=[];visible=[];oracle=[]
    for seed in ids:
        r=motion.episode(int(seed),steps=context+horizon,
                         size=size,balls=particles)
        positions=np.asarray(r["positions"],np.float32)/(size+16)
        missing=np.zeros(positions.shape[:2],bool)
        if occlude:
            missing[context-2:context+2,0]=True
        frames=render(positions,size,missing)
        observed,mask=observe(frames,particles)
        sequences.append(observed)
        visible.append(mask)
        oracle.append(positions)
    return np.stack(sequences),np.stack(visible),np.stack(oracle)


def run(out="runs/roadmap-39",seeds=(11,23,41),steps=30,batch=2,
        heldout=12,context=5,horizon=3,size=24,particles=2,channels=6):
    if not seeds or steps<1 or batch<1 or context<2 or horizon<1:
        raise ValueError("Invalid particle training protocol")
    if particles<1 or particles>4:
        raise ValueError("Synthetic palette supports 1..4 particles")
    folder=Path(out) if out else None
    if folder:folder.mkdir(parents=True,exist_ok=True)
    cases=[]
    for seed in seeds:
        init=nca.init(jax.random.PRNGKey(int(seed)),channels)
        ids=np.random.default_rng(seed+31).integers(31000,51000,(steps,batch))
        test_ids=np.arange(91000+seed*100,91000+seed*100+heldout)
        rgb,mask,truth=pools(test_ids,context,horizon,size,particles,True)
        # Truth for scoring only. The oracle input arm has explicit privilege.
        for representation in ("rgb_centroids","privileged_positions"):
            for messages in (True,False):
                params=init
                mm=jax.tree.map(jnp.zeros_like,params)
                vv=jax.tree.map(jnp.zeros_like,params)
                @jax.jit
                def update(p,m,v,examples,i):
                    value,grad=jax.value_and_grad(nca.loss)(
                        p,examples,context,horizon,messages)
                    norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grad)))
                    grad=jax.tree.map(lambda g:g/jnp.maximum(norm,1.),grad)
                    p,m,v=adamw(p,grad,m,v,i)
                    return p,m,v
                for i,group in enumerate(ids,1):
                    derived,_,known=pools(group,context,horizon,size,particles)
                    train=derived if representation=="rgb_centroids" else known
                    params,mm,vv=update(params,mm,vv,jnp.asarray(train),i)
                prefix=rgb if representation=="rgb_centroids" else truth
                predicted=np.asarray(nca.rollout(
                    params,jnp.asarray(prefix[:,:context]),horizon,messages))
                reference=task.controls(prefix[:,:context],horizon)
                metrics={key:task.score(value,truth[:,context:],
                                         (1,horizon),size-1)
                         for key,value in {"particle_nca":predicted,**reference}.items()}
                key=f"{representation}-{'messages' if messages else 'isolated'}"
                digest=None
                if folder:
                    checkpoint=folder/f"seed-{seed}-{key}.npz"
                    np.savez_compressed(checkpoint,**jax.tree.map(np.asarray,params))
                    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                cases.append({"seed":int(seed),"input":representation,
                              "messages":messages,"scores":metrics,
                              "checkpoint_sha256":digest,
                              "visible_fraction":float(mask[:,:context].mean())})
    report={"roadmap_id":39,"seeds":list(map(int,seeds)),
            "updates_per_arm":steps,"context":context,"horizon":horizon,
            "rows":cases,"limits":"RGB centroids use a known palette and per-object "
                     "colors, not learned object detection. Privileged arm receives "
                     "simulator positions; do NOT compare its MSE to RGB models as "
                     "equal-input evidence. Occluded positions are evaluator-only "
                     "for RGB. Message toggles share parameter count."}
    if folder:(folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-39")
    p.add_argument("--steps",type=int,default=30)
    a=p.parse_args()
    print(json.dumps({"rows":len(run(a.out,steps=a.steps)["rows"])}))


if __name__=="__main__":
    main()
