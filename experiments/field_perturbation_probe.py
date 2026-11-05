"""Pure physical impulse controls; optionally compare a frozen field model."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from src import field_perturbations as probes


def run(out="runs/field-perturbation/metrics.json",kind="diffusion",
        seeds=(11,23,41),steps=20,size=24,checkpoint=None,arm="cellular"):
    if not seeds or steps<2:
        raise ValueError("Missing seeds or steps")
    checks=probes.calibrate(seeds[0],size,steps)
    rows=[]
    for seed in seeds:
        controls=probes.match_controls(seed,kind,size,steps)
        row={"seed":int(seed),"positive_control_l2":controls["center"]["response_l2"].tolist(),
             "shifted_control_l2":controls["shifted"]["response_l2"].tolist(),
             "zero_control_l2":0.,
             "spread_radius2":controls["center"]["mean_squared_radius"].tolist(),
             "peak_response":controls["center"]["response_max_abs"].tolist()}
        if checkpoint is not None:
            from src import field_predictors as predictors
            with np.load(checkpoint,allow_pickle=False) as archive:
                params={k:jnp.asarray(archive[k]) for k in archive.files}
            # Isolated comparison, no training or future data in inference.
            def predict(start):
                frames=start[None]
                if kind=="diffusion":
                    frames=np.concatenate([frames,np.zeros_like(frames)],axis=1)
                frames=np.moveaxis(frames,1,-1)
                return np.asarray(predictors.rollout(
                    params,jnp.asarray(frames),steps-1,arm,kind))[0]
            truth=controls["center"]["response"][1:]
            before=controls["center"]["baseline"][0]
            intervened=before+controls["center"]["response"][0]
            model_response=predict(intervened)-predict(before)
            physical=np.moveaxis(
                np.concatenate([truth,np.zeros_like(truth)],axis=1)
                if kind=="diffusion" else truth,1,-1)
            row["model_perturbation_response_mse"]=float(
                np.mean((model_response-physical)**2))
            row["model_response_l2"]=float(np.linalg.norm(model_response))
        rows.append(row)
    report={"protocol":"calibrated-field-perturbations-v1",
            "kind":kind,"seeds":list(map(int,seeds)),"steps":steps,
            "calibration":checks,"rows":rows,"checkpoint_evaluated":checkpoint is not None,
            "limits":"Physical impulse response calibrated to a numerical "
             "reference. Same-energy displacement null has a different "
             "location, not an independent matched-response trajectory. "
             "Model-dependent claims require explicitly frozen checkpoints."}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=("wave","diffusion"),default="diffusion")
    p.add_argument("--out",default="runs/field-perturbation/metrics.json")
    p.add_argument("--checkpoint")
    p.add_argument("--arm",choices=("cellular","cnn"),default="cellular")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,23,41])
    a=p.parse_args()
    print(json.dumps(run(a.out,a.kind,a.seeds,checkpoint=a.checkpoint,
                         arm=a.arm)["calibration"],indent=2))


if __name__=="__main__":
    main()
