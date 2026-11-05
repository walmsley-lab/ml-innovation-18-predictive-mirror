"""Frozen two-frame vs full-history Gaussian observer and reset controls."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import (world3d, world3d_benchmark, rgb_gaussian_nca as short,
                 persistent_gaussian_observer as full,
                 gaussian_observer_occlusion as occlusion,
                 gaussian_slot_matching)
from experiments.gaussian_active_view import load
from experiments.gaussian_reconstruction_diagnostics import _png


def _center_error(record,trace,context):
    """Permutation-aware *evaluator-only* 3D world-center error over history."""
    costs=[]
    for t in range(context):
        truth=record["centers_world"][t]
        estimates=np.asarray(trace["centers"][t])
        matrix=np.linalg.norm(truth[:,None]-estimates[None],axis=-1)
        match,_=gaussian_slot_matching.assignment(matrix,miss_penalty=100.)
        costs.append(float(np.mean([cost for _,_,cost in match]))
                     if match else None)
    return costs


def run(out="runs/persistent-gaussian-observer",checkpoint=None,
        untrained_smoke=False,seeds=(11,23,41),context=7,horizon=5,size=24,
        hidden_lengths=(2,4)):
    if (not seeds or context<5 or horizon<1 or size<12 or
            (checkpoint is None and not untrained_smoke)):
        raise ValueError("Frozen checkpoint or explicit smoke needed")
    params=(load(checkpoint) if checkpoint else
            short.init(jax.random.PRNGKey(9),3,8))
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        record=world3d.episode(int(seed),context+horizon,size,"ood_camera")
        for hidden in hidden_lengths:
            if hidden>=context:
                raise ValueError("Occlusion longer than available history")
            observed,coverage=occlusion.occlude(record,context,hidden)
            # Truth future remains unmodified; hidden geometry is never
            # supplied to either predictor.
            views=jnp.asarray(observed[:context][None])
            actions=jnp.asarray(record["camera_actions"][None])
            K=jnp.asarray(record["K"])
            reset_context=np.asarray(views).copy()
            reset_context[:,:-1]=reset_context[:,-1:]
            shuffled_context=np.asarray(views).copy()
            shuffled_context[:,:-1]=shuffled_context[:,:-1][:,::-1]
            modes={
                "short_context":np.asarray(short.rollout(
                    params,views,actions,K,horizon,"cellular"))[0],
                "persistent":np.asarray(full.rollout(
                    params,views,actions,K,horizon,"cellular"))[0],
                "midpoint_reset":np.asarray(full.rollout(
                    params,views,actions,K,horizon,"cellular",
                    reset_at=context//2))[0],
                "final_frame_reset":np.asarray(full.rollout(
                    params,views,actions,K,horizon,"cellular",
                    reset_at=context-1))[0],
                "shuffled_history":np.asarray(full.rollout(
                    params,jnp.asarray(shuffled_context),actions,K,horizon,
                    "cellular"))[0],
            }
            score={name:world3d_benchmark.score(
                [record],values[None],context,tuple(range(1,horizon+1)))
                for name,values in modes.items()}
            center,velocity,scale,color,opacity,trace=full.observe(
                params,jnp.asarray(observed[:context]),
                jnp.asarray(record["camera_actions"][:context-1]),K)
            drift=_center_error(record,trace,context)
            display_indices=[0,min(2,horizon-1),horizon-1]
            panel=np.concatenate(
                [record["rgb"][context-1],
                 *[record["rgb"][context+i] for i in display_indices],
                 *[modes["short_context"][i] for i in display_indices],
                 *[modes["persistent"][i] for i in display_indices]],axis=1)
            image=f"seed-{seed}-hidden{hidden}-short-vs-persistent.png"
            _png(folder/image,panel)
            rows.append({"seed":int(seed),"hidden_observed_frames":int(hidden),
                         "coverage":coverage,"scores":score,
                         "per_frame_oracle_matched_center_error_m":drift,
                         "drift_last_minus_first_m":(
                             drift[-1]-drift[0]
                             if drift[0] is not None and drift[-1] is not None else None),
                         "gallery":image})
    report={"protocol":"persistent-gaussian-observer-v1",
            "checkpoint_sha256":(hashlib.sha256(
                Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None),
            "untrained_smoke":bool(untrained_smoke),
            "seeds":list(map(int,seeds)),"context":context,
            "horizon":horizon,"size":size,"rows":rows,
            "modes":["short_context","persistent","midpoint_reset",
                     "final_frame_reset","shuffled_history"],
            "limits":"No added training in this probe. All arms share "
                     "the same frozen RGB-to-slot encoder and cellular "
                     "dynamics, but the persistent observer adds a hand-coded "
                     "confidence-weighted center/velocity filter. Occluder "
                     "location uses simulator truth ONLY when generating "
                     "observed training/test RGB, never passed to model. "
                     "Drift is evaluator-only oracle matching, not inferred "
                     "metric 3D truth. Unknown camera actions are NOT tested. "
                     "A result is not evidence of learned biological memory."}
    (folder/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/persistent-gaussian-observer")
    p.add_argument("--checkpoint")
    p.add_argument("--untrained-smoke",action="store_true")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--context",type=int,default=7)
    p.add_argument("--horizon",type=int,default=5)
    p.add_argument("--size",type=int,default=24)
    p.add_argument("--hidden-lengths",nargs="+",type=int,default=[2,4])
    a=p.parse_args()
    m=run(a.out,a.checkpoint,a.untrained_smoke,a.seeds,a.context,
          a.horizon,a.size,a.hidden_lengths)
    print(json.dumps({"rows":len(m["rows"]),"evaluated":m["modes"]},indent=2))


if __name__=="__main__":
    main()
