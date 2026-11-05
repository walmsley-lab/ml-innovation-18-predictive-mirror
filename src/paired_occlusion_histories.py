"""Matched-last-frame occlusion worlds: history identifies hidden velocity.

Two examples share the EXACT last observed RGB frame and occluder but
have opposing observed approach trajectories and different futures.
This prevents last-frame-only predictors from passing the benchmark.
"""
import numpy as np
from src.rgb_motion_world import COLORS

CONTEXT=8
HORIZON=10


def episode(seed=11,direction=1,context=CONTEXT,horizon=HORIZON,size=32):
    if direction not in (-1,1) or context<8 or horizon<6 or size<32:
        raise ValueError("Need a direction and context>=8, horizon>=6, size>=32")
    rng=np.random.default_rng(int(seed))
    speed=float(rng.choice([1.85,2.,2.15]))
    cy=float(size//2+rng.integers(-2,3))
    mid=(size-1)/2
    radius=2.25
    xx,yy=np.meshgrid(np.arange(size),np.arange(size))
    stripe=np.abs(xx-mid)<=6.0
    frames,labels,positions=[],[],[]
    for t in range(context+horizon):
        x=mid+direction*speed*(t-(context-1))
        disc=(xx-x)**2+(yy-cy)**2<=radius**2
        visible=disc&~stripe
        count=int(np.count_nonzero(visible))
        total=int(np.count_nonzero(disc))
        if count==0:
            label="barrier" if total>0 else "out_of_view"
        elif count<total:
            label="partial"
        else:
            label="visible"
        frame=np.full((size,size,3),.035,np.float32)
        frame[stripe]=[.13,.13,.15]
        frame[visible]=COLORS[0]
        frames.append(frame);labels.append(label);positions.append((x,cy))
    if labels[context-1]!="barrier":
        raise AssertionError("Last observed image must fully hide moving object")
    events=[]
    for t in range(1,context+horizon):
        if labels[t] in ("partial","visible") and labels[t-1]=="barrier":
            start=t-1
            while start>0 and labels[start-1]=="barrier":
                start-=1
            events.append({"frame":t,"kind":"barrier_return","hidden_start":start})
    events=[e for e in events if context<=e["frame"]<context+horizon]
    if len(events)!=1 or not any(l=="visible" for l in labels[:context]):
        raise AssertionError("Paired world lacks a guaranteed meaningful return")
    return {"frames":np.stack(frames),"visible":np.asarray(
                [l in ("visible","partial") for l in labels])[:,None],
            "visibility_reason":np.asarray(labels),
            "events":events,"seed":int(seed),"mode":"paired_barrier",
            "color":COLORS[0].copy(),
            "positions":np.asarray(positions,np.float32),
            "context":context,"horizon":horizon,"direction":int(direction),
            "speed":speed}


def pair(seed=11,context=CONTEXT,horizon=HORIZON,size=32):
    positive=episode(seed,1,context,horizon,size)
    negative=episode(seed,-1,context,horizon,size)
    np.testing.assert_array_equal(
        positive["frames"][context-1],negative["frames"][context-1])
    if np.array_equal(positive["frames"][:context-1],
                      negative["frames"][:context-1]):
        raise AssertionError("Histories must differ before final observation")
    if np.array_equal(positive["frames"][context:],
                      negative["frames"][context:]):
        raise AssertionError("Expected futures must differ")
    return positive,negative


def controls(contexts):
    """Reset and shuffled histories retain the EXACT same final frame."""
    x=np.asarray(contexts)
    if x.ndim!=5 or x.shape[1]<3:
        raise ValueError("Expected [batch,context,H,W,RGB]")
    reset=np.broadcast_to(x[:,-1:,:,:,:],x.shape).copy()
    shuffled=np.concatenate([x[:,:-1][:,::-1],x[:,-1:]],axis=1)
    return {"intact":x,"reset":reset,"shuffled":shuffled}


def linear_history_forecast(context_rgb,horizon=HORIZON):
    """Simple non-neural history control: detect visible colored disc, fit x(t).

    Assumes benchmark's known palette/background/occluder geometry but
    does not access simulator positions, future RGB or object state.
    """
    seen=[]
    color=COLORS[0]
    for t,frame in enumerate(context_rgb):
        mask=np.max(np.abs(frame-color),axis=-1)<.025
        if mask.sum()>=6:
            yy,xx=np.where(mask)
            # Visible only, prior to occlusion: fit early, not tiny edge slivers.
            seen.append((t,float(xx.mean()),float(yy.mean()),int(mask.sum())))
    if len(seen)<2:
        return np.repeat(context_rgb[-1:,:,:,:],horizon,axis=0)
    # First three largest unoccluded discs avoid partial-mask centroid drift.
    stable=sorted(seen,key=lambda r:(-r[3],r[0]))[:3]
    stable=sorted(stable)
    x=np.asarray([p[0] for p in stable],float)
    xp=np.asarray([p[1] for p in stable],float)
    yp=np.asarray([p[2] for p in stable],float)
    velocity=float(np.polyfit(x,xp,1)[0]) if len(np.unique(x))>1 else 0.
    intercept=float(np.mean(xp-velocity*x))
    cy=float(np.mean(yp))
    radius=2.25  # Declared known benchmark primitive prior.
    size=context_rgb.shape[1]
    mid=(size-1)/2
    yy,xx=np.mgrid[:size,:size]
    occluder=np.abs(xx-mid)<=6.0
    frames=[]
    for t in range(len(context_rgb),len(context_rgb)+horizon):
        cx=intercept+velocity*t
        frame=context_rgb[-1].copy()
        mask=((xx-cx)**2+(yy-cy)**2<=radius**2)&~occluder
        frame[mask]=color
        frames.append(frame)
    return np.stack(frames)
