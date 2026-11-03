"""3D active-view policy from frozen ensemble forecast disagreement.

The policy cannot access scene geometry, positions, depth, or future RGB.
All policies are scored from a standardized final camera viewpoint.
"""
import numpy as np
import jax.numpy as jnp
from src import world3d, action_rgb_nca as model
from src.active_gaze_comparison import paired_ci

CANDIDATES=np.array([[-.12,0.],[0.,0.],[.12,0.],
                     [0.,-.015],[0.,.015]],np.float32)


def forecast(params,frames,actions,next_action):
    length=min(3,len(frames))
    if length<2:
        raise ValueError("Need at least two observed views")
    history=np.asarray(frames[-length:],np.float32)[None]
    planned=np.asarray([*actions[-(length-1):],next_action],np.float32)[None]
    return np.asarray(model.rollout(
        params,jnp.asarray(history),jnp.asarray(planned),1,True))[0,0]


def select_action(ensemble,frames,actions):
    """Argmax forecast disagreement is a *heuristic*, not information gain."""
    if len(frames)<2:
        return np.zeros(2,np.float32)
    proposals=[]
    for action in CANDIDATES:
        predicted=np.stack([forecast(p,frames,actions,action) for p in ensemble])
        disagreement=float(np.var(predicted,axis=0).mean())
        proposals.append(disagreement-.00001*float(np.sum(action*action)))
    return CANDIDATES[int(np.argmax(proposals))].copy()


def evaluate_episode(seed,ensemble,policy="active",steps=5,size=24):
    if policy not in ("fixed","random","active") or steps<3 or not ensemble:
        raise ValueError("Invalid sensing policy/ensemble")
    # Action-invariant hidden world; policy sees render() output only.
    record=world3d.episode(seed,steps+1,size,"ood_depth",
                          actions=np.zeros((steps,2),np.float32))
    xyz=np.zeros(3,np.float32)
    yaw=0.
    frames,actions,seen=[],[],set()
    rng=np.random.default_rng(seed+99000)
    for t in range(steps):
        image,_,ids=world3d.render(
            record["centers_world"][t],record["radii"],record["colors"],
            xyz,yaw,record["K"],size)
        frames.append(image)
        seen.update(int(x) for x in np.unique(ids) if x>=0)
        if t<steps-1:
            if policy=="active":
                choice=select_action(ensemble,frames,actions)
            elif policy=="random":
                choice=CANDIDATES[int(rng.integers(len(CANDIDATES)))].copy()
            else:
                choice=np.zeros(2,np.float32)
            actions.append(choice)
            xyz[0]+=choice[0]
            yaw+=float(choice[1])
    # All policies are tested at the same *home* view and same world time.
    return_home=np.array([-xyz[0],-yaw],np.float32)
    prediction=np.mean(np.stack([
        forecast(p,frames,actions,return_home) for p in ensemble]),axis=0)
    target,_,ids=world3d.render(
        record["centers_world"][steps],record["radii"],record["colors"],
        np.zeros(3,np.float32),0.,record["K"],size)
    square=np.mean((prediction-target)**2,axis=-1)
    mask=ids>=0
    return {"seed":int(seed),"policy":policy,
            "home_view_mse":float(square.mean()),
            "home_view_visible_mse":float(square[mask].mean()) if np.any(mask) else None,
            "observed_instance_count":len(seen),
            "observations":steps,"actions":np.asarray(actions).tolist(),
            "predicted_home_rgb":prediction,
            "actual_home_rgb":target}


def compare(ensemble,seeds=(11,23,37),steps=5,size=24):
    rows={policy:[evaluate_episode(seed,ensemble,policy,steps,size)
                  for seed in seeds]
          for policy in ("fixed","random","active")}
    summary={}
    for policy,records in rows.items():
        summary[policy]={
            "mean_home_mse":float(np.mean([x["home_view_mse"] for x in records])),
            "mean_seen_instances":float(np.mean([x["observed_instance_count"]
                                                for x in records])),
            "mean_home_visible_mse":float(np.mean([
                x["home_view_visible_mse"] for x in records
                if x["home_view_visible_mse"] is not None]))
                    if any(x["home_view_visible_mse"] is not None for x in records)
                    else None}
    comparison={}
    for control in ("fixed","random"):
        comparison[f"active_minus_{control}"]=paired_ci(
            [x["home_view_mse"] for x in rows["active"]],
            [x["home_view_mse"] for x in rows[control]])
    return {"protocol":"active-calibrated-world3d-v1","seeds":list(map(int,seeds)),
            "observations":steps,"scores":summary,"paired":comparison,
            "limits":"Disagreement is an uncalibrated heuristic; source models "
                     "must be trained and frozen for scientific comparison. "
                     "The final standardized home-view forecast avoids "
                     "choosing easy views, but does not establish metric "
                     "depth inference or actual information gain."}
