"""Causal interception game; future truth never enters a policy."""
import numpy as np

def trajectory(seed=1,steps=20,occluded=(7,12)):
    rng=np.random.default_rng(seed)
    start=float(rng.uniform(3,7));speed=float(rng.uniform(-.25,.25))
    y=np.clip(start+np.arange(steps)*speed,0,10)
    observations=[None if occluded[0]<=t<occluded[1] else float(y[t])
                  for t in range(steps)]
    return {"observations":observations,"evaluator_only":y,
            "occlusion":occluded,"seed":seed}

def reactive(history):
    visible=[x for x in history if x is not None]
    return visible[-1] if visible else 5.

def motion_estimate(history,remaining):
    points=[(t,float(x)) for t,x in enumerate(history) if x is not None]
    if len(points)<2:return reactive(history)
    (t0,y0),(t1,y1)=points[-2:]
    speed=(y1-y0)/(t1-t0)
    return float(np.clip(y1+speed*(len(history)-1-t1+remaining),0,10))

def evaluate(policy,episodes,decision_time=10,paddle_width=.5):
    if not episodes or decision_time<2:raise ValueError("Invalid evaluation")
    rows=[]
    for ep in episodes:
        history=ep["observations"][:decision_time]
        remaining=len(ep["observations"])-decision_time
        location=float(policy(history,remaining))
        truth=float(ep["evaluator_only"][-1])
        rows.append({"error":abs(location-truth),
                     "hit":abs(location-truth)<=paddle_width})
    return {"hit_rate":float(np.mean([r["hit"] for r in rows])),
            "mean_error":float(np.mean([r["error"] for r in rows])),
            "episodes":len(rows)}

def reactive_policy(history,remaining):
    return reactive(history)

def velocity_policy(history,remaining):
    return motion_estimate(history,remaining)
