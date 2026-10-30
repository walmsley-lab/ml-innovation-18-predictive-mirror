"""Active sensing with a known tiny 1D world: no reinforcement learning."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


WORLD = 20
FIELD = 7
START = 6
ACTIONS = (-3, 0, 3)


def observe(object_x, view):
    """Local pixel offset, or -1 when the object is not in the sensed crop."""
    return object_x - view if view <= object_x < view + FIELD else -1


def posterior(belief, view, result):
    """Exact Bayesian filter for a deterministic measurement model."""
    likelihood = np.array([observe(x, view) == result for x in range(WORLD)])
    updated = belief * likelihood
    total = updated.sum()
    return updated / total if total else belief.copy()


def risk(belief):
    """Expected squared error of posterior-mean position estimate."""
    indices = np.arange(WORLD)
    mean = float(belief @ indices)
    return float(belief @ (indices - mean) ** 2)


def expected_risk(belief, view):
    """Expected posterior risk before a prospective look (no truth access)."""
    return sum(float(belief[indices].sum()) * risk(posterior(belief, view, result))
               for result in range(-1, FIELD)
               if (indices := np.array([x for x in range(WORLD) if observe(x, view) == result])).size)


def choose_gaze(belief, view):
    """Deterministic greedy selection minimizing expected posterior risk."""
    choices = [(expected_risk(belief, int(np.clip(view + a, 0, WORLD - FIELD))),
                abs(a), a) for a in ACTIONS]
    return min(choices)[2]


def follow(object_x, policy, budget=5, seed=0, fixed_actions=None):
    """Execute a policy without passing hidden object_x to the action policy."""
    rng = np.random.default_rng(seed)
    belief = np.full(WORLD, 1.0 / WORLD)
    view = START
    observations = []
    for t in range(budget):
        result = observe(object_x, view)
        belief = posterior(belief, view, result)
        observations.append((int(view), int(result)))
        if t == budget - 1:
            break
        action = (0 if policy == "fixed" else
                  int(rng.choice(ACTIONS)) if policy == "random" else
                  choose_gaze(belief, view) if policy == "greedy" else
                  fixed_actions[t])
        view = int(np.clip(view + action, 0, WORLD - FIELD))
    return float(belief @ np.arange(WORLD)), float(risk(belief)), observations


def optimized_fixed_actions(budget=5):
    """Exhaustive static sensing schedule against a uniform world prior."""
    candidates = itertools.product(ACTIONS, repeat=budget - 1)
    return min(candidates, key=lambda actions: sum(
        follow(x, "optimized", budget, fixed_actions=actions)[1] for x in range(WORLD)))


def compare(budget=5):
    fixed_best = optimized_fixed_actions(budget)
    results = {}
    for policy in ("fixed", "random", "optimized", "greedy"):
        errors, risks = [], []
        for x in range(WORLD):
            estimate, remaining_risk, _ = follow(
                x, policy, budget, seed=1000 + x, fixed_actions=fixed_best)
            errors.append(abs(x - estimate))
            risks.append(remaining_risk)
        results[policy] = {"mean_abs_position_error": float(np.mean(errors)),
                           "expected_posterior_variance": float(np.mean(risks)),
                           "observation_budget": budget}
    results["optimized_schedule"] = list(fixed_best)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--out", default="runs/gaze.json")
    args = parser.parse_args()
    if not 2 <= args.budget <= 7:
        parser.error("Budget must be 2..7 to keep exhaustive baseline small")
    result = compare(args.budget)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
