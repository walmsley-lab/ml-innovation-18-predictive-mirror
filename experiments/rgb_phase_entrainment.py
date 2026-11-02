"""Four-arm RGB phase-entrainment study, separate from earlier phase ablation."""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import predictive_rgb_nca as base, rgb_phase_entrainment as phase
from src import rgb_benchmark as bench
from train import adamw


def run(out="runs/rgb-phase-entrainment", seeds=(11, 23, 37), steps=40,
        context=8, horizon=10, size=24, ticks=2, channels=8):
    if not seeds or steps < 1 or context < 2 or horizon < 1 or ticks < 1:
        raise ValueError("Invalid phase experiment settings")
    train = bench.frames(bench.scenes(31000, 12, context, horizon, size))
    validation = bench.frames(bench.scenes(61000, 4, context, horizon, size))
    test = bench.frames(bench.scenes(71000, 4, context, horizon, size))
    leads = tuple(sorted({horizon, *(h for h in bench.LEADS if h <= horizon)}))
    controls = {name: bench.score(value, test, context, leads)
                for name, value in bench.controls(test[:, :context], horizon).items()}
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        initial = base.init(jax.random.PRNGKey(int(seed)), channels)
        def validation_mse(params, mode):
            pred = np.asarray(phase.rollout(
                params, jnp.asarray(validation[:, :context]), horizon, ticks, mode))
            return float(np.square(pred-validation[:, context:]).mean())
        def trainer(mode):
            @jax.jit
            def update(p, m, v, examples, step):
                error, grads = jax.value_and_grad(
                    lambda w: phase.loss(w, examples, context, horizon, ticks, mode))(p)
                norm = jnp.sqrt(sum(jnp.square(g).sum() for g in jax.tree.leaves(grads)))
                grads = jax.tree.map(lambda g: g/jnp.maximum(norm, 1.), grads)
                p, m, v = adamw(p, grads, m, v, step)
                return p, m, v, error
            return update
        updates = {mode: trainer(mode) for mode in phase.MODES}
        arms = {}
        for mode in phase.MODES:
            zero = validation_mse(initial, mode)
            arms[mode] = {"params": initial,
                          "m": jax.tree.map(jnp.zeros_like, initial),
                          "v": jax.tree.map(jnp.zeros_like, initial),
                          "best": initial, "step": 0, "best_val": zero,
                          "step_zero_val": zero}
        rng = np.random.default_rng(int(seed) + 1000)
        for step in range(1, steps+1):
            x = jnp.asarray(train[rng.integers(len(train), size=2)])
            for mode, state in arms.items():
                state["params"], state["m"], state["v"], _ = updates[mode](
                    state["params"], state["m"], state["v"], x, step)
            if step == steps or step % max(1, steps//5) == 0:
                for mode, state in arms.items():
                    metric = validation_mse(state["params"], mode)
                    if metric < state["best_val"]:
                        state["best"], state["step"], state["best_val"] = (
                            state["params"], step, metric)
        report_arms = {}
        for mode, state in arms.items():
            before = time.perf_counter()
            prediction, final_phase = phase.rollout(
                state["best"], jnp.asarray(test[:, :context]),
                horizon, ticks, mode, True)
            forecast = np.asarray(prediction)
            # Explicitly synchronize CPU evaluation for wall time.
            _ = float(np.square(forecast).sum())
            elapsed = (time.perf_counter()-before)*1000
            report_arms[mode] = {
                "selected_step": state["step"],
                "step_zero_validation_mse": state["step_zero_val"],
                "selected_validation_mse": state["best_val"],
                "test": bench.score(forecast, test, context, leads),
                "phase_global_order": float(phase.phase_order(final_phase)),
                "batch_eval_wall_ms_includes_compilation": elapsed}
            np.savez_compressed(target/f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray, state["best"]))
        results.append({"seed": int(seed), "arms": report_arms})
    report = {"experiment": "rgb_phase_sensory_entrainment",
              "modes": list(phase.MODES), "seeds": list(map(int, seeds)),
              "steps": steps, "context": context, "horizon": horizon,
              "baselines": controls, "results": results,
              "limitations": "Oscillator phase adds nontrainable state and runtime. "
                             "Entrainment uses sensory prediction error only when "
                             "a real context observation arrives; no future peeking. "
                             "Global phase order is not visual-binding evidence. "
                             "First-call inference times include JAX compilation. "
                             "Step zero is eligible in all arms."}
    (target/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/rgb-phase-entrainment")
    parser.add_argument("--seeds",type=int,nargs="+",default=[11,23,37])
    parser.add_argument("--steps",type=int,default=40)
    parser.add_argument("--context",type=int,default=8)
    parser.add_argument("--horizon",type=int,default=10)
    parser.add_argument("--size",type=int,default=24)
    parser.add_argument("--ticks",type=int,default=2)
    parser.add_argument("--channels",type=int,default=8)
    a=parser.parse_args()
    r=run(a.out,a.seeds,a.steps,a.context,a.horizon,a.size,a.ticks,a.channels)
    print(json.dumps({"runs":len(r["results"]),"output":str(Path(a.out)/"metrics.json")},indent=2))


if __name__=="__main__":
    main()
