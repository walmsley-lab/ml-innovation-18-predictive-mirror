"""Small observable-control lab for filtering, smoothing and latent memory.

This is a synthetic hidden Markov model, NOT a trained RGB neural network.
Online forecasts only see the past. Backward smoothing is diagnostic only.
"""
import argparse
import json

import numpy as np


TRANSITION = np.array([
    [0.78, 0.19, 0.03],
    [0.02, 0.83, 0.15],
    [0.24, 0.04, 0.72],
], dtype=float)
P_ONE = np.array([0.12, 0.50, 0.88], dtype=float)


def stationary(transition=TRANSITION):
    p = np.ones(len(transition)) / len(transition)
    for _ in range(10000):
        q = p @ transition
        if np.max(np.abs(q - p)) < 1e-14:
            return q
        p = q
    raise RuntimeError("no stationary convergence")


def reverse_kernel(transition=TRANSITION, pi=None):
    """R[current,previous] from stationary Bayes; NOT an inverse transition."""
    pi = stationary(transition) if pi is None else pi
    return (pi[:, None] * transition).T / pi[:, None]


def emission_probability(observation, one=P_ONE):
    return one if observation == 1 else 1 - one


def filter_sequence(observations, transition=TRANSITION, one=P_ONE):
    """Beliefs P(z_t | y_0,...,y_t), strictly causal."""
    prior = stationary(transition)
    beliefs = []
    for step, y in enumerate(observations):
        if step:
            prior = beliefs[-1] @ transition
        posterior = prior * emission_probability(int(y), one)
        posterior /= posterior.sum()
        beliefs.append(posterior)
    return np.array(beliefs)


def smooth_sequence(observations, transition=TRANSITION, one=P_ONE):
    """Retrospective P(z_t | all observations), never valid for a live forecast."""
    filtered = filter_sequence(observations, transition, one)
    beta = np.ones(len(transition))
    smoothed = np.zeros_like(filtered)
    for t in range(len(observations) - 1, -1, -1):
        smoothed[t] = filtered[t] * beta
        smoothed[t] /= smoothed[t].sum()
        if t:
            beta = transition @ (emission_probability(int(observations[t]), one) * beta)
            beta /= beta.sum()
    return smoothed


def next_one_probability(belief, transition=TRANSITION, one=P_ONE):
    return float((belief @ transition) @ one)


def sample_episode(rng, steps, transition=TRANSITION, one=P_ONE):
    state = int(rng.choice(len(transition), p=stationary(transition)))
    observations = []
    for _ in range(steps):
        observations.append(int(rng.random() < one[state]))
        state = int(rng.choice(len(transition), p=transition[state]))
    return observations


def cross_entropy(prob, label):
    p = float(np.clip(prob, 1e-12, 1 - 1e-12))
    return -np.log(p if label else 1 - p)


def evaluate(episodes=200, steps=20, seed=0):
    """Score predictions issued at t against y[t+1]; only prefix y[:t+1]."""
    if episodes < 1 or steps < 4:
        raise ValueError("episodes >= 1 and steps >= 4 required")
    rng = np.random.default_rng(seed)
    pi = stationary()
    sums = dict(history=0.0, last_observation=0.0, shuffled_past=0.0, unconditional=0.0)
    count = 0
    for _ in range(episodes):
        ys = sample_episode(rng, steps)
        beliefs = filter_sequence(ys)
        for t in range(2, steps - 1):
            prefix = ys[:t + 1]
            target = ys[t + 1]
            last_only = pi * emission_probability(ys[t])
            last_only /= last_only.sum()
            shuffled = list(prefix[:-1])
            rng.shuffle(shuffled)
            shuffled.append(prefix[-1])
            predictions = {
                "history": next_one_probability(beliefs[t]),
                "last_observation": next_one_probability(last_only),
                "shuffled_past": next_one_probability(filter_sequence(shuffled)[-1]),
                "unconditional": float(pi @ P_ONE),
            }
            for name, probability in predictions.items():
                sums[name] += cross_entropy(probability, target)
            count += 1
    r = reverse_kernel()
    return {
        "task": "synthetic noisy three-state HMM, next binary observation",
        "episodes": episodes, "steps": steps, "seed": seed, "forecasts": count,
        "nll_nats_per_forecast": {k: float(v / count) for k, v in sums.items()},
        "reverse_rows_sum_error": float(np.max(np.abs(r.sum(axis=1) - 1))),
        "status": "analytic oracle-control experiment; no learned RGB or NCA evidence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.episodes, args.steps, args.seed), indent=2))


if __name__ == "__main__":
    main()
