"""Matched-home-view 3D active sensing using RGB-inferred Gaussian models.

Viewpoint selection sees RGB observations and known action commands only.
Hidden world state is consulted *only* to render the next camera image and
compute a standardized future-home-view evaluation target.
"""
import numpy as np
import jax.numpy as jnp
from src import rgb_gaussian_nca,world3d
from src.active_gaze_comparison import paired_ci

CANDIDATES=np.asarray([[-.12,0],[0,0],[.12,0],
                       [0,-.02],[0,.02]],np.float32)
POLICIES=("fixed","random","active","yoked")


def predict_one(params,history,actions,next_action,K):
    if len(history)<2 or len(actions)!=len(history)-1:
        raise ValueError("History/actions must align")
    observed=jnp.asarray(np.stack(history)[None])
    commanded=jnp.asarray(np.stack([*actions,next_action])[None])
    return np.asarray(rgb_gaussian_nca.rollout(
        params,observed,commanded,jnp.asarray(K),1,"cellular"))[0,0]


def select(ensemble,history,actions,K,candidates=CANDIDATES):
    """Ensemble disagreement is a heuristic—not calibrated information gain."""
    if len(history)<2 or len(ensemble)<2:
        return np.zeros(2,np.float32),0
    values=[]
    for action in candidates:
        forecasts=np.stack([predict_one(p,history,actions,action,K)
                            for p in ensemble])
        disagreement=float(np.var(forecasts,axis=0).mean())
        values.append(disagreement-1e-6*float(action@action))
    return np.asarray(candidates[np.argmax(values)]).copy(),len(candidates)*len(ensemble)


def episode(seed,ensemble,policy,steps=5,size=24,domain="ood_depth",
            yoked_actions=None):
    if policy not in POLICIES or steps<3 or size<12 or len(ensemble)<2:
        raise ValueError("Invalid sensing episode")
    if policy=="yoked" and (yoked_actions is None or
                             np.asarray(yoked_actions).shape!=(steps-1,2)):
        raise ValueError("Yoked policy requires donor action sequence")
    # World motion does not depend on the camera plan; same seed -> same
    # objects and future home-view target across all sensing policies.
    world=world3d.episode(seed,steps+1,size,domain,
                          actions=np.zeros((steps,2),np.float32))
    K=world["K"]
    x,yaw=0.,0.
    history,actions=[],[]
    probes=0
    rng=np.random.default_rng(int(seed)+190000)
    for t in range(steps):
        frame,_,_=world3d.render(
            world["centers_world"][t],world["radii"],world["colors"],
            np.array([x,0,0],np.float32),yaw,K,size)
        history.append(frame)
        if t==steps-1:
            break
        if t==0:
            # Nobody has a two-frame belief yet; all policies take same action.
            action=np.zeros(2,np.float32)
        elif policy=="active":
            action,cost=select(ensemble,history,actions,K)
            probes+=cost
        elif policy=="random":
            action=CANDIDATES[int(rng.integers(len(CANDIDATES)))].copy()
        elif policy=="yoked":
            action=np.asarray(yoked_actions[t],np.float32).copy()
        else:
            action=np.zeros(2,np.float32)
        actions.append(action)
        x+=float(action[0]);yaw+=float(action[1])
    home=np.array([-x,-yaw],np.float32)
    estimated=np.mean(np.stack([
        predict_one(p,history,actions,home,K) for p in ensemble]),axis=0)
    actual,_,labels=world3d.render(
        world["centers_world"][steps],world["radii"],world["colors"],
        np.zeros(3,np.float32),0.,K,size)
    mse=np.square(estimated-actual).mean(axis=-1)
    visible=labels>=0
    return {
        "seed":int(seed),"policy":policy,
        "home_mse":float(mse.mean()),
        "home_visible_mse":float(mse[visible].mean()) if np.any(visible) else None,
        "actions":[a.tolist() for a in actions],
        "observations":steps,
        "selection_forecast_calls":probes,
        "mean_action_magnitude":float(np.linalg.norm(actions,axis=-1).mean()),
    }


def compare(ensemble,seeds=(11,23,41),domains=("ood_depth","ood_camera"),
            steps=5,size=24):
    if not seeds or not domains or len(ensemble)<2:
        raise ValueError("Need held-out worlds and ensemble")
    result={}
    for domain in domains:
        runs=[]
        for index,seed in enumerate(seeds):
            donor=int(seeds[(index+1)%len(seeds)]) if len(seeds)>1 else int(seed)+99991
            donor_actions=episode(donor,ensemble,"active",steps,size,domain)["actions"]
            donor_actions=np.asarray(donor_actions,np.float32)
            policies={name:episode(
                int(seed),ensemble,name,steps,size,domain,
                yoked_actions=donor_actions if name=="yoked" else None)
                for name in POLICIES}
            runs.append({"seed":int(seed),"policies":policies,
                         "donor_seed":donor})
        means={name:float(np.mean([
                row["policies"][name]["home_mse"] for row in runs]))
               for name in POLICIES}
        active=np.asarray([row["policies"]["active"]["home_mse"] for row in runs])
        comparisons={name:paired_ci(
            active,np.asarray([row["policies"][name]["home_mse"] for row in runs]))
            for name in ("fixed","random","yoked")}
        result[domain]={"mean_home_mse":means,
                        "active_minus_control":comparisons,"episodes":runs}
    return {"protocol":"rgb-gaussian-active-3d-v1",
            "domains":result,"seeds":list(map(int,seeds)),
            "observations_per_policy":steps,
            "limits":"The same number of camera observations are used, "
             "but active selection makes additional model calls and is "
             "therefore NOT compute-matched. Ensemble disagreement is "
             "not calibrated uncertainty. Yoked actions come from an "
             "independent donor world and are replayed on the target. "
             "A shared final-home viewpoint prevents easy-view selection "
             "from masquerading as better prediction. No actual exploration "
             "gain claimed without trained, frozen checkpoints."}
