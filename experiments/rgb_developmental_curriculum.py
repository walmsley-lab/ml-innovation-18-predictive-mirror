"""Matched staged versus shuffled RGB motion curriculum.

Both arms see the exact same examples, multiset and update budget.
Only the order of presentation differs; test and validation remain frozen.
"""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src import predictive_rgb_nca as nca, rgb_benchmark as bench
from train import adamw


STAGES = ("simple", "occlusion", "interaction")
WORLD = {
    "simple": {"balls": 1, "occlusion": False, "camera_motion": False},
    "occlusion": {"balls": 1, "occlusion": True, "camera_motion": False},
    "interaction": {"balls": 2, "occlusion": True, "camera_motion": True},
}


def curriculum_orders(steps, seed):
    """Two permutations of the exact same indexed training examples."""
    if steps < 3 or steps % len(STAGES):
        raise ValueError("steps must be a positive multiple of three")
    labels = np.repeat(np.arange(len(STAGES)), steps//len(STAGES))
    order = np.arange(steps)
    shuffled = np.random.default_rng(int(seed)+17000).permutation(steps)
    return {"staged": order, "shuffled": shuffled}, labels


def make_batches(steps, batch_size, context, horizon, size):
    """Pre-render once; both arms are presented the same concrete episodes."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    _, labels = curriculum_orders(steps, 0)
    bank = []
    for index, label in enumerate(labels):
        stage = STAGES[int(label)]
        records = bench.scenes(
            31000 + index*batch_size, batch_size, context, horizon, size,
            **WORLD[stage])
        bank.append(bench.frames(records))
    return np.stack(bank), labels


def separate_splits(context, horizon, size, episodes):
    """Different seed ranges and scenarios for selection and final test."""
    val, test = {}, {}
    for index, stage in enumerate(STAGES):
        val[stage] = bench.frames(bench.scenes(
            61000+index*1000, episodes, context, horizon, size, **WORLD[stage]))
        test[stage] = bench.frames(bench.scenes(
            71000+index*1000, episodes, context, horizon, size, **WORLD[stage]))
    return val, test


def run(out="runs/rgb-developmental-curriculum", seeds=(11, 23, 37),
        steps=30, batch_size=2, heldout=6, context=8, horizon=10,
        size=24, ticks=2, channels=8):
    if (not seeds or steps < 3 or batch_size < 1 or heldout < 1
            or context < 2 or horizon < 1 or ticks < 1 or channels < 1):
        raise ValueError("Invalid training/evaluation settings")
    bank, labels = make_batches(steps, batch_size, context, horizon, size)
    validation, test = separate_splits(context, horizon, size, heldout)
    leads = tuple(sorted({horizon, *(h for h in bench.LEADS if h <= horizon)}))
    controls = {
        stage: {name: bench.score(forecast, test[stage], context, leads)
                for name, forecast in bench.controls(
                    test[stage][:, :context], horizon).items()}
        for stage in STAGES}
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        initial = nca.init(jax.random.PRNGKey(int(seed)), channels)
        orders, _ = curriculum_orders(steps, seed)

        def validation_mse(params):
            errors = []
            for stage in STAGES:
                observed = validation[stage]
                pred = np.asarray(nca.rollout(
                    params, jnp.asarray(observed[:, :context]), horizon, ticks))
                errors.append(float(np.square(
                    pred-observed[:, context:context+horizon]).mean()))
            return float(np.mean(errors))

        @jax.jit
        def update(p, m, v, batch, step):
            error, gradients = jax.value_and_grad(
                lambda w: nca.loss(w, batch, context, horizon, ticks))(p)
            norm = jnp.sqrt(sum(jnp.square(g).sum()
                                for g in jax.tree.leaves(gradients)))
            gradients = jax.tree.map(lambda g: g/jnp.maximum(1., norm), gradients)
            p, m, v = adamw(p, gradients, m, v, step)
            return p, m, v, error

        start = validation_mse(initial)
        arms = {
            mode: {"params": initial,
                   "m": jax.tree.map(jnp.zeros_like, initial),
                   "v": jax.tree.map(jnp.zeros_like, initial),
                   "best": initial, "best_step": 0,
                   "best_val": start, "step_zero_val": start}
            for mode in orders}
        for step in range(1, steps+1):
            for mode, state in arms.items():
                example_index = int(orders[mode][step-1])
                state["params"], state["m"], state["v"], _ = update(
                    state["params"], state["m"], state["v"],
                    jnp.asarray(bank[example_index]), step)
            if step == steps or step % max(1, steps//5) == 0:
                for mode, state in arms.items():
                    value = validation_mse(state["params"])
                    if value < state["best_val"]:
                        state["best"], state["best_val"], state["best_step"] = (
                            state["params"], value, step)

        arm_results = {}
        for mode, state in arms.items():
            scores = {}
            example_predictions = {}
            for stage in STAGES:
                observations = test[stage]
                forecast = np.asarray(nca.rollout(
                    state["best"], jnp.asarray(observations[:, :context]),
                    horizon, ticks))
                scores[stage] = bench.score(forecast, observations, context, leads)
                example_predictions[stage] = forecast[0]
            arm_results[mode] = {
                "selected_step": state["best_step"],
                "step_zero_validation_mse": state["step_zero_val"],
                "selected_validation_mse": state["best_val"],
                "test_by_scenario": scores,
                "presentation_stage_order": [
                    STAGES[int(labels[index])] for index in orders[mode]]}
            np.savez_compressed(folder/f"seed-{seed}-{mode}.npz",
                                **jax.tree.map(np.asarray, state["best"]))
            np.savez_compressed(
                folder/f"seed-{seed}-{mode}-example.npz",
                observed=test["interaction"][0, :context],
                actual=test["interaction"][0, context:],
                predicted=example_predictions["interaction"])
        results.append({"seed": int(seed), "arms": arm_results})

    report = {
        "experiment": "rgb_nca_order_only_developmental_curriculum",
        "seeds": list(map(int, seeds)), "steps": steps, "batch_size": batch_size,
        "context": context, "horizon": horizon,
        "stages": list(STAGES), "scenarios": WORLD,
        "train_unique_batch_count": len(bank),
        "same_training_multiset_per_arm": True,
        "validation_test_episodes_per_stage": heldout,
        "baselines_by_scenario": controls, "results": results,
        "limitations": "Exactly the same training episode batches and steps per arm; "
                       "only presentation order differs. Three stage families use "
                       "separate deterministic train/validation/test seed ranges. "
                       "Validation averages all scenarios and selects the checkpoint, "
                       "including step zero. No claim of emergent developmental stages "
                       "or transfer to real video without independent evaluation."}
    (folder/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/rgb-developmental-curriculum")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,23,37])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heldout", type=int, default=6)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    a = parser.parse_args()
    result = run(a.out, a.seeds, a.steps, a.batch_size, a.heldout,
                 a.context, a.horizon, a.size, a.ticks, a.channels)
    print(json.dumps({"runs": len(result["results"]),
                      "metrics": str(Path(a.out)/"metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
