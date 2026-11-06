"""Can small two-tower embeddings recognize a supplied visual future?

This is candidate *recognition*, not forecasting: the test-time candidate set
contains the correct continuation by construction. No hidden state/velocity is
passed to any scorer. Distinct scene seeds are used for each data split.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from src import baselines
from src.multifuture_motion import render

DIRECTIONS = np.asarray(((0, 1), (0, -1), (1, 0), (-1, 0)), np.float32)


def scene(seed, context=8, horizon=4, size=32):
    """All four continuations use the same renderer and initial observation."""
    if context < 3 or horizon < 1 or size < 24:
        raise ValueError("context >=3, horizon >=1 and size >=24 required")
    rng = np.random.default_rng(int(seed))
    center = rng.uniform(size * .43, size * .57, size=2)
    speed = float(rng.uniform(.95, 1.25))
    direction = int(rng.integers(4))
    history = np.stack([render(center + (t - context + 1) * speed
                               * DIRECTIONS[direction], size)
                        for t in range(context)])
    choices = np.stack([np.stack([render(center + (t + 1) * speed * d, size)
                                  for t in range(horizon)])
                        for d in DIRECTIONS])
    order = rng.permutation(4)
    return history, choices[order], int(np.flatnonzero(order == direction)[0])


def dataset(first_seed, count, context=8, horizon=4, size=32):
    if count < 1:
        raise ValueError("count must be positive")
    records = [scene(first_seed + i, context, horizon, size)
               for i in range(count)]
    return (np.stack([r[0] for r in records]),
            np.stack([r[1] for r in records]),
            np.asarray([r[2] for r in records], np.int32))


def features(videos):
    """Observation-only RGB->coarse appearance; first/middle/last frames.

    Inputs have shape [...,T,H,W,3]. No simulator truth or future labels.
    """
    x = np.asarray(videos, np.float32)
    if x.ndim < 4 or x.shape[-1] != 3 or x.shape[-3] % 8 or x.shape[-2] % 8:
        raise ValueError("Expected RGB videos with spatial dimensions divisible by 8")
    frames = x[..., [0, x.shape[-4] // 2, -1], :, :, :].max(axis=-1)
    h, w = frames.shape[-2:]
    pooled = frames.reshape(*frames.shape[:-2], 8, h // 8, 8, w // 8).mean(axis=(-3, -1))
    return ((pooled - .035) * 10).reshape(*x.shape[:-4], 3 * 8 * 8).astype(np.float32)


def initialize(key, dim=192, width=32):
    a, b, c, d = jax.random.split(key, 4)
    return {"state_w": jax.random.normal(a, (dim, width)) * .08,
            "state_b": jnp.zeros(width),
            "future_w": jax.random.normal(b, (dim, width)) * .08,
            "future_b": jnp.zeros(width),
            "pair_w": jax.random.normal(c, (dim * 2, width)) * .08,
            "pair_b": jnp.zeros(width),
            "pair_out": jax.random.normal(d, (width,)) * .08}


def tower_scores(params, states, futures, temperature=.1, in_batch=False):
    """Cosine similarities, with cached candidate embeddings at inference."""
    state = jnp.tanh(states @ params["state_w"] + params["state_b"])
    state = state / (jnp.linalg.norm(state, axis=-1, keepdims=True) + 1e-8)
    candidate = jnp.tanh(futures @ params["future_w"] + params["future_b"])
    candidate = candidate / (jnp.linalg.norm(candidate, axis=-1, keepdims=True) + 1e-8)
    if in_batch:
        return state @ candidate.reshape(-1, candidate.shape[-1]).T / temperature
    return jnp.einsum("bd,bkd->bk", state, candidate) / temperature


def pair_scores(params, states, futures):
    repeated = jnp.broadcast_to(states[:, None, :], futures.shape)
    pair = jnp.concatenate((repeated, futures), axis=-1)
    return jnp.einsum("bkw,w->bk", jnp.tanh(pair @ params["pair_w"]
                          + params["pair_b"]), params["pair_out"])


def loss(params, states, futures, labels, mode):
    if mode == "contrastive":
        logits = tower_scores(params, states, futures, in_batch=True)
        targets = jnp.arange(states.shape[0]) * futures.shape[1] + labels
    elif mode == "supervised_mlp":
        logits = pair_scores(params, states, futures)
        targets = labels
    else:
        raise ValueError(mode)
    return -jnp.mean(jax.nn.log_softmax(logits, axis=-1)[jnp.arange(len(labels)), targets])


def fit(initial, train, mode, steps=100, batch=16, seed=17, lr=.004):
    if steps < 0 or batch < 1:
        raise ValueError("steps >=0 and batch >=1 required")
    s, f, y = map(jnp.asarray, train)
    # Different arms have disjoint trainable parameters; no unused large head.
    keys = ("state_w", "state_b", "future_w", "future_b") if mode == "contrastive" else (
        "pair_w", "pair_b", "pair_out")
    params = {key: initial[key] for key in keys}
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)

    @jax.jit
    def update(p, m, v, indices, step):
        value, g = jax.value_and_grad(loss)(p, s[indices], f[indices], y[indices], mode)
        m = jax.tree.map(lambda old, grad: .9 * old + .1 * grad, m, g)
        v = jax.tree.map(lambda old, grad: .999 * old + .001 * grad**2, v, g)
        p = jax.tree.map(lambda weight, a, b: weight - lr * (a / (1 - .9**step)) /
                         (jnp.sqrt(b / (1 - .999**step)) + 1e-8), p, m, v)
        return p, m, v, value

    rng = np.random.default_rng(seed)
    last = float(loss(params, s[:batch], f[:batch], y[:batch], mode))
    for step in range(1, steps + 1):
        idx = jnp.asarray(rng.integers(len(y), size=batch))
        params, m, v, value = update(params, m, v, idx, step)
        last = float(value)
    return params, last


def ranks(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.ndim != 2 or scores.shape[0] != len(labels):
        raise ValueError("scores must have shape [examples, candidates]")
    return {"accuracy": float(np.mean(scores.argmax(axis=1) == labels)),
            "margin": float(np.mean(scores[np.arange(len(labels)), labels] -
                                    np.max(np.where(np.eye(scores.shape[1])[labels].astype(bool),
                                                    -np.inf, scores), axis=1)))}


def pixel_controls(histories, choices):
    """Compare supplied futures to persistence and observed constant velocity."""
    rows = {name: [] for name in ("persistence", "constant_velocity")}
    for past, options in zip(histories, choices):
        forecasts = {"persistence": baselines.persistence_forecast(past, len(options[0])),
                     "constant_velocity": baselines.constant_velocity_forecast(past, len(options[0]))}
        for name, prediction in forecasts.items():
            rows[name].append(-np.mean((options - prediction[None]) ** 2,
                                        axis=(1, 2, 3, 4)))
    return {name: np.asarray(values) for name, values in rows.items()}


def run(out="runs/contrastive-future/metrics.json", seeds=(11, 23, 41),
        steps=100, train_count=128, test_count=32, context=8, horizon=4):
    if not seeds or train_count < 2 or test_count < 2:
        raise ValueError("Need seeds and train/test scenes")
    start = time.perf_counter()
    train_raw = dataset(31000, train_count, context, horizon)
    test_raw = dataset(91000, test_count, context, horizon)
    training = (features(train_raw[0]), features(train_raw[1]), train_raw[2])
    testing = (features(test_raw[0]), features(test_raw[1]), test_raw[2])
    baseline = {name: ranks(values, test_raw[2])
                for name, values in pixel_controls(test_raw[0], test_raw[1]).items()}
    reports = []
    for seed in seeds:
        initial = initialize(jax.random.PRNGKey(int(seed)))
        random_scores = tower_scores(initial, jnp.asarray(testing[0]),
                                     jnp.asarray(testing[1]))
        arms = {"untrained_two_tower": ranks(random_scores, testing[2])}
        for mode in ("contrastive", "supervised_mlp"):
            trained, final_loss = fit(initial, training, mode, steps, seed=int(seed))
            logits = (tower_scores(trained, jnp.asarray(testing[0]),
                                   jnp.asarray(testing[1])) if mode == "contrastive"
                      else pair_scores(trained, jnp.asarray(testing[0]),
                                       jnp.asarray(testing[1])))
            arms[mode] = {**ranks(logits, testing[2]),
                          "train_last_batch_loss": final_loss,
                          "parameters": int(sum(np.size(w) for w in trained.values()))}
        reports.append({"seed": int(seed), "arms": arms})
    report = {"experiment": "contrastive_future_recognition_v1",
              "data": {"train_seeds": [31000, 31000 + train_count - 1],
                       "test_seeds": [91000, 91000 + test_count - 1],
                       "train": train_count, "test": test_count,
                       "context": context, "horizon": horizon,
                       "choices": len(DIRECTIONS)},
              "steps": steps, "controls": baseline, "runs": reports,
              "elapsed_seconds": time.perf_counter() - start,
              "limitations": "Oracle-included candidate recognition, not future generation. "
                 "Correct continuation is always supplied; four deterministic cardinal "
                 "motions use the same renderer. No real camera or ambiguous branch inference; "
                 "softmax scores are not calibrated probabilities. Test-only scenes never "
                 "enter training. Compare held-out accuracy to classical constant velocity."}
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/contrastive-future/metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 41])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=32)
    a = parser.parse_args()
    report = run(a.out, a.seeds, a.steps, a.train_count, a.test_count)
    print(json.dumps({"controls": report["controls"], "runs": report["runs"]}, indent=2))


if __name__ == "__main__":
    main()
