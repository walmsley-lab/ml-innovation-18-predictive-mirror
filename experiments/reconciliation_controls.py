"""Exploratory nulls for perceptual reconciliation; no hidden labels during network training."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.reconciliation import fit_probe, decode
from src import model, multisheet
from src.world import sequence


def within_sheet_lesion(params, seed=71):
    """Zero as many within-sheet kernel weights as there are cross-sheet weights."""
    rng = np.random.default_rng(seed)
    matrices = {k: np.asarray(v).copy() for k, v in params.items()}
    eligible = []
    for name, channel_start, channel_end in (("fast", 2, 8), ("slow", 2, 4)):
        for idx in np.ndindex(matrices[name].shape):
            if channel_start <= idx[2] < channel_end:
                eligible.append((name, idx))
    # 9*(2*6+6*2) = 216 cross-sheet weights, including unused zeros.
    selected = rng.choice(len(eligible), size=216, replace=False)
    for index in selected:
        name, position = eligible[int(index)]
        matrices[name][position] = 0
    return {name: jnp.asarray(v) for name, v in matrices.items()}


def single_trace(params, frames, actions):
    h, w = frames.shape[1:3]
    hidden = jnp.zeros((1, h, w, 8), jnp.float32)
    def update(state, inputs):
        frame, action = inputs
        state, _ = model.step(params, state, frame[None], action)
        return state, state[0]
    _, traces = jax.lax.scan(update, hidden, (frames[:-1], actions))
    return traces


def features(params, seeds, architecture="multisheet", mode="intact", means=None):
    """Post-hoc x position labels and actual barrier-occlusion mask are evaluator only."""
    first, second, targets, occluded = [], [], [], []
    for seed in seeds:
        frames, actions, truth = sequence(int(seed), steps=12)
        if architecture == "single":
            history = np.asarray(single_trace(params, frames, actions))
            a, b = history[..., :6], history[..., 6:]
        else:
            _, a, b = multisheet.trace(params, frames, actions,
                                       feedback_strength=0.0 if mode == "cut" else 1.0,
                                       mean_messages=means if mode == "mean" else None)
            a, b = np.asarray(a), np.asarray(b)
        first.extend(a.mean(axis=1).reshape(12, -1))
        second.extend(b.mean(axis=1).reshape(12, -1))
        targets.extend(truth["world_x"][1:] - truth["camera_x"][1:])
        occluded.extend(truth["occluded"][1:])
    return tuple(map(np.asarray, (first, second, targets, occluded)))


def score(readouts, probe_a, probe_b):
    a = decode(readouts[0], probe_a)
    b = decode(readouts[1], probe_b)
    truth, occluded = readouts[2], readouts[3]
    compatible = np.abs(a - b) <= 1
    accurate = (np.abs(a - truth) <= 1) & (np.abs(b - truth) <= 1)
    return {
        "occluded_frames": int(occluded.sum()),
        "compatible_fraction": float(np.mean(compatible[occluded])) if occluded.any() else None,
        "accurate_and_compatible_fraction": float(np.mean((compatible & accurate)[occluded]))
        if occluded.any() else None,
        "decoder_a_mae_px": float(np.mean(np.abs(a - truth))),
        "decoder_b_mae_px": float(np.mean(np.abs(b - truth))),
    }


def estimate_means(params, seeds):
    fast, slow = [], []
    for seed in seeds:
        frames, actions, _ = sequence(int(seed), steps=12)
        _, a, b = multisheet.trace(params, frames, actions)
        fast.append(np.asarray(a).mean(axis=(0, 1, 2)))
        slow.append(np.asarray(b).mean(axis=(0, 1, 2)))
    return jnp.asarray(np.mean(fast, axis=0)), jnp.asarray(np.mean(slow, axis=0))


def compare(multi_params, single_params, probe_seeds, test_seeds, no_feedback_params=None):
    probe_seeds, test_seeds = tuple(probe_seeds), tuple(test_seeds)
    if not probe_seeds or not test_seeds or set(probe_seeds) & set(test_seeds):
        raise ValueError("Probe and test seeds must be independent")
    training = features(multi_params, probe_seeds)
    probe_a = fit_probe(training[0], training[2])
    probe_b = fit_probe(training[1], training[2])
    means = estimate_means(multi_params, probe_seeds)
    result = {
        "intact": score(features(multi_params, test_seeds), probe_a, probe_b),
        "cut_at_test": score(features(multi_params, test_seeds, mode="cut"), probe_a, probe_b),
        "mean_feedback": score(features(multi_params, test_seeds, mode="mean", means=means),
                               probe_a, probe_b),
        "within_sheet_lesion": score(features(within_sheet_lesion(multi_params), test_seeds),
                                    probe_a, probe_b),
    }
    single_train = features(single_params, probe_seeds, architecture="single")
    wa, wb = fit_probe(single_train[0], single_train[2]), fit_probe(single_train[1], single_train[2])
    result["single_sheet_split_null"] = score(
        features(single_params, test_seeds, architecture="single"), wa, wb)
    if no_feedback_params is not None:
        # Decoders for this trained control must be fitted independently.
        baseline_train = features(no_feedback_params, probe_seeds, mode="cut")
        ca, cb = fit_probe(baseline_train[0], baseline_train[2]), fit_probe(baseline_train[1], baseline_train[2])
        result["trained_without_feedback"] = score(
            features(no_feedback_params, test_seeds, mode="cut"), ca, cb)
    return result


def load(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: jnp.asarray(file[key]) for key in file.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multisheet", required=True)
    parser.add_argument("--single", required=True)
    parser.add_argument("--no-feedback")
    parser.add_argument("--probe-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--out", default="runs/reconciliation-controls.json")
    args = parser.parse_args()
    if args.probe_episodes < 2 or args.test_episodes < 1:
        parser.error("Not enough independent episodes")
    report = compare(load(args.multisheet), load(args.single),
                     range(40_000_000, 40_000_000 + args.probe_episodes),
                     range(50_000_000, 50_000_000 + args.test_episodes),
                     load(args.no_feedback) if args.no_feedback else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
