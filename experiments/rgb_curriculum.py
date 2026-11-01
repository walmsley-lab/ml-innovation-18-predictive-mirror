"""Paired one-second RGB NCA training: fixed scenes vs diverse scene curriculum.

Keeps model and optimizer fixed. Evaluations use disjoint validation/test
identities; oracle world state never reaches the model.
"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_benchmark as benchmark
from train import adamw

SCENARIOS = (
    {"balls": 1},
    {"balls": 1, "occlusion": True},
    {"balls": 1, "camera_motion": True},
    {"balls": 2},
)


def batch(start_seed, count, context, horizon, size, scenario):
    records = benchmark.scenes(start_seed, count, context, horizon, size,
                               **SCENARIOS[scenario])
    return benchmark.frames(records)


def evaluate(params, samples, context, horizon, ticks):
    predicted = np.asarray(nca.rollout(params, jnp.asarray(samples[:, :context]),
                                       horizon, ticks))
    return float(np.square(predicted - samples[:, context:]).mean())


def run(output="runs/rgb-curriculum", seeds=(11, 23, 41), steps=100,
        batch_size=2, context=8, horizon=10, ticks=2, size=32, channels=12):
    if not seeds or steps < 1 or batch_size < 1 or context < 2 or horizon < 1:
        raise ValueError("Invalid training schedule")
    # Validation and final test contain every scenario; identities do not overlap.
    validation = np.concatenate([batch(61000+100*i, 3, context, horizon, size, i)
                                 for i in range(len(SCENARIOS))])
    test = np.concatenate([batch(71000+100*i, 3, context, horizon, size, i)
                           for i in range(len(SCENARIOS))])
    controls = benchmark.controls(test[:, :context], horizon)

    @jax.jit
    def update(p, m, v, x, step):
        error, grads = jax.value_and_grad(
            lambda weights: nca.loss(weights, x, context, horizon, ticks))(p)
        norm = jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
        grads = jax.tree.map(lambda g: g / jnp.maximum(norm, 1), grads)
        p, m, v = adamw(p, grads, m, v, step)
        return p, m, v, error

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        initial = nca.init(jax.random.PRNGKey(int(seed)), channels)
        zero = lambda: jax.tree.map(jnp.zeros_like, initial)
        arms = {name: {"p": initial, "m": zero(), "v": zero(),
                       "best": initial, "step": 0,
                       "val": evaluate(initial, validation, context, horizon, ticks)}
                for name in ("single", "curriculum")}
        history = []
        for step in range(1, steps+1):
            # Paired initialization, same update budget, same RNG root.
            # Single arm sees a fixed single trajectory; curriculum arm
            # sees fresh identities and progressively harder world families.
            stage = min(len(SCENARIOS)-1, (step-1)*len(SCENARIOS)//steps)
            x = {
                "single": batch(31000+int(seed), 1, context, horizon, size, 0),
                "curriculum": batch(32000+int(seed)*100000+step*batch_size,
                                    batch_size, context, horizon, size, stage),
            }
            x["single"] = np.repeat(x["single"], batch_size, axis=0)
            for name, arm in arms.items():
                arm["p"], arm["m"], arm["v"], error = update(
                    arm["p"], arm["m"], arm["v"], jnp.asarray(x[name]), step)
            if step == steps or step % max(1, steps//5) == 0:
                row = {"step": step, "stage": stage}
                for name, arm in arms.items():
                    metric = evaluate(arm["p"], validation, context, horizon, ticks)
                    row[name+"_validation_mse"] = metric
                    if metric < arm["val"]:
                        arm["best"], arm["val"], arm["step"] = arm["p"], metric, step
                history.append(row)
                print(f"seed={seed} step={step} stage={stage} "
                      f"single={row['single_validation_mse']:.6f} "
                      f"curriculum={row['curriculum_validation_mse']:.6f}",
                      flush=True)

        entry = {"seed": int(seed), "arms": {}, "history": history}
        leads = tuple(sorted({horizon, *(h for h in benchmark.LEADS if h <= horizon)}))
        for name, arm in arms.items():
            forecast = np.asarray(nca.rollout(arm["best"],
                                  jnp.asarray(test[:, :context]), horizon, ticks))
            entry["arms"][name] = {
                "best_step": int(arm["step"]), "validation_mse": arm["val"],
                "metrics": benchmark.score(forecast, test, context, leads)}
            np.savez_compressed(out/f"seed-{seed}-{name}.npz",
                                **jax.tree.map(np.asarray, arm["best"]))
        results.append(entry)
    lead = str(horizon)
    baseline_scores = {name: benchmark.score(values, test, context, (horizon,))
                       for name, values in controls.items()}
    report = {"experiment":"paired_rgb_trajectory_curriculum",
              "seeds":list(map(int,seeds)), "steps":steps, "horizon":horizon,
              "context":context, "world_families":len(SCENARIOS),
              "validation_seed_roots":[61000+100*i for i in range(4)],
              "test_seed_roots":[71000+100*i for i in range(4)],
              "controls":baseline_scores,"runs":results,
              "summary":{name:{
                  "mean_one_second_mse":float(np.mean([
                      r["arms"][name]["metrics"][lead]["mse"] for r in results])),
                  "step_zero_count":sum(r["arms"][name]["best_step"]==0
                                        for r in results),
                  "beats_velocity_count":sum(
                      r["arms"][name]["metrics"][lead]["mse"] <
                      baseline_scores["constant_velocity"][lead]["mse"]
                      for r in results)}
                  for name in ("single","curriculum")},
              "caveat":"This compares learning curricula, not new architectures. "
                       "Optical flow and independent webcam transfer remain separate."}
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="runs/rgb-curriculum")
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,41])
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--context",type=int,default=8)
    p.add_argument("--horizon",type=int,default=10)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--size",type=int,default=32)
    p.add_argument("--channels",type=int,default=12)
    a=p.parse_args()
    r=run(a.out,a.seeds,a.steps,a.batch_size,a.context,a.horizon,
          a.ticks,a.size,a.channels)
    print(json.dumps(r["summary"],indent=2))


if __name__=="__main__":
    main()
