"""Paired, same-world evaluation of active versus passive gaze controls."""
import numpy as np
from src import active_gaze_dynamics as gaze


def paired_ci(first,second,seed=19,bootstraps=300):
    first,second=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    if first.shape!=second.shape or first.ndim!=1 or not len(first):
        raise ValueError("Need paired 1D series")
    delta=first-second
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(delta),size=(bootstraps,len(delta)))
    low,high=np.quantile(delta[draws].mean(axis=1),(.025,.975))
    return {"mean_difference":float(delta.mean()),
            "episode_bootstrap_95pct":[float(low),float(high)],
            "pairs":len(delta)}


def compare(seeds=(11,23,37),episodes=24,context=8,horizon=10):
    if not seeds or episodes<1 or context<2 or horizon<1:
        raise ValueError("Invalid active-gaze comparison")
    ids=[int(seed)*100000+i for seed in seeds for i in range(episodes)]
    policies=("fixed","random","active")
    records={policy:[gaze.episode(index,policy,context,horizon)
                     for index in ids] for policy in policies}
    for i in range(len(ids)):
        targets={records[p][i]["target_future_x"] for p in policies}
        if len(targets)!=1:
            raise AssertionError("Policies did not share hidden world")
    metrics={}
    for policy,rows in records.items():
        aware=np.asarray([r["predictions"]["action_aware"]["absolute_error"] for r in rows])
        blind=np.asarray([r["predictions"]["action_blind"]["absolute_error"] for r in rows])
        entropy=np.asarray([r["predictions"]["action_aware"]["forecast_entropy_nats"] for r in rows])
        observed=np.asarray([sum(v["local_observation"]>=0 for v in r["views"])/context for r in rows])
        metrics[policy]={
            "aware_position_mae":float(aware.mean()),
            "blind_position_mae":float(blind.mean()),
            "aware_entropy_nats":float(entropy.mean()),
            "visible_observation_fraction":float(observed.mean()),
            "episodes":len(rows),
        }
    active=np.asarray([r["predictions"]["action_aware"]["absolute_error"]
                       for r in records["active"]])
    comparisons={}
    for baseline in ("fixed","random"):
        errors=np.asarray([r["predictions"]["action_aware"]["absolute_error"]
                           for r in records[baseline]])
        comparisons[f"active_minus_{baseline}"]=paired_ci(active,errors)
    return {
        "experiment":"paired_active_gaze_control_evidence",
        "seeds":list(map(int,seeds)),"episodes_per_seed":episodes,
        "context":context,"horizon":horizon,
        "metrics":metrics,"paired_comparisons":comparisons,
        "limits":"Known 1D state dynamics, NOT learned neural gaze, RGB, "
                 "metric 3D or independent webcam improvement. The same "
                 "hidden trajectory is used for each policy, but their "
                 "observations differ by action. Confidence intervals "
                 "bootstrap episodes conditional on this simulator. "
                 "Choosing a policy that sees more of the world is not "
                 "automatically proof of better world modeling.",
    }
