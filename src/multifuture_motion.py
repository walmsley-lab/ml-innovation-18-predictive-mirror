"""Minimal irreducibly ambiguous RGB motion world and coherent futures.

Two episodes have IDENTICAL observed histories and physically different
continuations. A single-point predictor cannot always be correct.
Branch labels and true future trajectories are evaluator-only.
"""
import numpy as np
from src.rgb_motion_world import COLORS


def render(center,size=32,radius=2.4):
    yy,xx=np.mgrid[:size,:size]
    frame=np.full((size,size,3),.035,np.float32)
    mask=(yy-center[0])**2+(xx-center[1])**2<=radius**2
    frame[mask]=COLORS[0]
    return frame


def episode(pair_seed=11,branch=0,context=5,horizon=6,size=32):
    if branch not in (0,1) or context<3 or horizon<1 or size<24:
        raise ValueError("Invalid fork episode")
    rng=np.random.default_rng(int(pair_seed))
    first=np.array([size*.5,float(size*.20)],np.float32)
    vx=float(rng.uniform(.75,1.15))
    speed=float(rng.uniform(.75,1.15))
    positions=[]
    for t in range(context+horizon):
        x=first[1]+vx*t
        y=first[0] if t<context else (
            first[0]+(1 if branch else -1)*speed*(t-context+1))
        positions.append((y,x))
    positions=np.asarray(positions,np.float32)
    return {"frames":np.stack([render(p,size) for p in positions]),
            "positions":positions,"branch":branch,"pair_seed":int(pair_seed),
            "context":context,"horizon":horizon,
            "branch_speed":speed}


def detect(frame):
    rgb=np.asarray(frame,np.float32)
    mask=np.max(np.abs(rgb-COLORS[0]),axis=-1)<.12
    if not mask.any():
        return None
    y,x=np.nonzero(mask)
    return np.array([y.mean(),x.mean()],np.float32)


def train_probabilities(episodes,smoothing=1):
    """Fit categorical branch rates on training-only labels.

    Does NOT learn to distinguish mutually indistinguishable futures.
    """
    count=np.asarray([sum(r["branch"]==0 for r in episodes),
                      sum(r["branch"]==1 for r in episodes)],np.float64)
    if count.sum()==0 or smoothing<0:
        raise ValueError("Need training episodes")
    return (count+smoothing)/(count.sum()+2*smoothing)


def predict(history,horizon=6,probabilities=(.5,.5),speed=1.0):
    """Return two coherent, open-loop future paths and rendered images."""
    history=np.asarray(history)
    if history.ndim!=4 or len(history)<3 or horizon<1 or speed<=0:
        raise ValueError("Past RGB history required")
    centers=[detect(f) for f in history[-2:]]
    if any(c is None for c in centers):
        raise ValueError("Object absent in final observed context")
    velocity=centers[1]-centers[0]
    probability=np.asarray(probabilities,np.float64)
    if probability.shape!=(2,) or np.any(probability<0) or (
            abs(probability.sum()-1)>1e-7):
        raise ValueError("Two normalized branch probabilities required")
    paths=[];videos=[]
    for sign in (-1,1):
        path=np.stack([centers[1]+(step+1)*velocity+
                       np.array([sign*speed*(step+1),0.],np.float32)
                       for step in range(horizon)])
        paths.append(path)
        videos.append(np.stack([render(c,history.shape[1]) for c in path]))
    return {"paths":np.stack(paths),
            "images":np.stack(videos),
            "probabilities":probability}


def energy_score(paths,truth,probabilities):
    """Proper multivariate energy score; smaller is better."""
    paths=np.asarray(paths,np.float64)
    target=np.asarray(truth,np.float64)
    weights=np.asarray(probabilities,np.float64)
    if paths.ndim!=3 or paths.shape[1:]!=target.shape or len(paths)!=len(weights):
        raise ValueError("Paths/target/probabilities mismatch")
    flattened=paths.reshape(len(paths),-1)
    target=target.reshape(-1)
    first=np.sum(weights*np.linalg.norm(flattened-target,axis=-1))
    diffs=np.linalg.norm(flattened[:,None]-flattened[None],axis=-1)
    second=.5*np.sum(weights[:,None]*weights[None,:]*diffs)
    return float(first-second)


def measures(pred,truth,label,coverage_px=2):
    paths=pred["paths"]
    probabilities=pred["probabilities"]
    distance=np.linalg.norm(paths-np.asarray(truth)[None],axis=-1)
    winning=int(np.argmax(probabilities))
    return {"energy_score":energy_score(paths,truth,probabilities),
            "top1_ade_px":float(distance[winning].mean()),
            "expected_ade_px":float(np.sum(probabilities*distance.mean(axis=-1))),
            "oracle_best_of_many_ade_px":float(distance.mean(axis=-1).min()),
            "all_frames_covered":bool(np.any(np.all(distance<=coverage_px,axis=1))),
            "per_frame_coverage":(distance<=coverage_px).any(axis=0).astype(int).tolist(),
            "trajectory_diversity_px":float(np.mean(np.linalg.norm(
                paths[0]-paths[1],axis=-1))),
            "branch_brier":float(np.sum((probabilities-
                 np.eye(2)[label])**2)),
            "effective_hypothesis_count":float(1/np.sum(probabilities**2))}


def coast(history,horizon):
    previous,current=[detect(frame) for frame in history[-2:]]
    if previous is None or current is None:raise ValueError("Missing observed object")
    v=current-previous
    return np.stack([current+(t+1)*v for t in range(horizon)])
