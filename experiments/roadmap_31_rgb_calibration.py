"""Roadmap 31 RGB bridge: conformalize already-issued pixel forecasts.

Frozen persistence, image-derived constant velocity or supplied RGB NCA
checkpoint; training is *not* performed here. Per-episode calibration targets
are future observed RGB; simulator positions/flow never enter the predictor.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src import rgb_motion_world as world, rgb_benchmark as benchmark
from experiments.roadmap_31_calibration import quantile

MODELS = ("persistence", "constant_velocity", "nca")


def frames(start_seed, count, context=8, horizon=5, size=24):
    if count < 10 or context < 2 or horizon < 1:
        raise ValueError("Need independent episodes and a valid window")
    records=[]
    for seed in range(start_seed,start_seed+count):
        record=world.episode(
            seed,steps=context+horizon,size=size,balls=1+(seed%2),
            occlusion=(seed%3==0),camera_motion=(seed%3==1))
        records.append(record["frames"])  # Only observed images retained.
    return np.stack(records)


def forecast(observed_context, horizon, model="persistence", checkpoint=None):
    if model not in MODELS or observed_context.ndim!=5 or horizon<1:
        raise ValueError("Invalid forecast settings")
    if model != "nca":
        return benchmark.controls(observed_context,horizon)[model]
    if not checkpoint:
        raise ValueError("NCA needs its own trained checkpoint")
    import jax.numpy as jnp
    from src import predictive_rgb_nca as nca
    with np.load(checkpoint) as saved:
        params={k:jnp.asarray(saved[k]) for k in saved.files}
    return np.asarray(nca.rollout(params,jnp.asarray(observed_context),
                                  horizon,ticks=2))


def abs_max_errors(prediction, batch, context):
    """One episode and lead are a unit; simultaneous over all RGB pixels."""
    truth=batch[:,context:context+prediction.shape[1]]
    if prediction.shape!=truth.shape:
        raise ValueError("Forecast and future RGB shape mismatch")
    return np.max(np.abs(prediction-truth),axis=(2,3,4))


def report_one(prediction,observed,context,radius):
    target=observed[:,context:context+prediction.shape[1]]
    residual=abs_max_errors(prediction,observed,context)
    covered=residual<=radius[None]
    scores=benchmark.score(prediction,observed,context,
                           tuple(range(1,prediction.shape[1]+1)))
    return {
        "all_pixels_per_lead_coverage":np.mean(covered,axis=0).tolist(),
        "mean_lead_coverage":float(covered.mean()),
        "all_leads_joint_coverage_diagnostic":float(np.all(covered,axis=1).mean()),
        "mean_channel_interval_width":float(np.mean(2*radius)),
        "rgb_mse":float(np.square(prediction-target).mean()),
        "changed_region_mse_last_lead":scores[str(prediction.shape[1])]["motion_mse"]}


def run(out="runs/roadmap-31-rgb/metrics.json",model="persistence",
        checkpoint=None,calibration_count=40,test_count=40,
        context=8,horizon=5,size=24,alpha=.1):
    if min(calibration_count,test_count)<10 or not 0<alpha<1:
        raise ValueError("Insufficient RGB episodes or invalid alpha")
    calibrated=frames(61000,calibration_count,context,horizon,size)
    test=frames(91000,test_count,context,horizon,size)
    predicted_cal=forecast(calibrated[:,:context],horizon,model,checkpoint)
    predicted_test=forecast(test[:,:context],horizon,model,checkpoint)
    residual=abs_max_errors(predicted_cal,calibrated,context)
    radii=np.array([quantile(residual[:,k],alpha) for k in range(horizon)])
    score=report_one(predicted_test,test,context,radii)
    fixed=np.full(horizon,.25)
    report={
        "roadmap_id":31,"protocol":"rgb-frozen-forecast-interval-v1",
        "model":model,"checkpoint":str(checkpoint) if checkpoint else None,
        "alpha":alpha,"calibration_episodes":calibration_count,
        "test_episodes":test_count,"context":context,"horizon":horizon,
        "size_px":size,"calibrated_radius_per_lead":radii.tolist(),
        "test":{"conformal":score,
                "fixed_radius_control":report_one(predicted_test,test,context,fixed)},
        "limits":"Simultaneous per-lead RGB pixel bounds (max-channel error), "
                 "NOT jointly calibrated across all leads. Whole episodes "
                 "are independent sampling units. Calibration can be "
                 "conservative; camera domain shift voids exchangeability. "
                 "No latent uncertainty is learned and checkpoint performance "
                 "is not established by this wrapper."}
    if out is not None:
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/roadmap-31-rgb/metrics.json")
    p.add_argument("--model",choices=MODELS,default="persistence")
    p.add_argument("--checkpoint")
    p.add_argument("--calibration-count",type=int,default=40)
    p.add_argument("--test-count",type=int,default=40)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.model,a.checkpoint,a.calibration_count,
                         a.test_count)["test"],indent=2))


if __name__=="__main__":
    main()
