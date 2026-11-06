"""Tiny embodied organization world: infer fit from vision and limited touch.

A physical relation is checked only when an agent probes or tries to nest two
items. Object geometry and the oracle solution remain evaluator-only.
This is a symbolic baseline, not a trained neural perception or robotics model.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

SHAPES = ("round", "square")
CLEARANCE = 0.055


def make_items(seed, count=6, noise=0.11):
    """Construct a seeded partially observable world of nestable containers."""
    if not 2 <= count <= 7 or noise < 0:
        raise ValueError("Expected 2-7 items and nonnegative visual noise")
    rng = random.Random(seed)
    items = []
    for index in range(count):
        scale = rng.uniform(0.3, 0.93)
        width = max(0.12, scale + rng.uniform(-0.14, 0.14))
        depth = max(0.12, scale + rng.uniform(-0.17, 0.17))
        items.append({
            "id": index, "shape": rng.choice(SHAPES),
            "true_width": width, "true_depth": depth,
            "seen_width": max(0.08, width + rng.gauss(0, noise)),
            "seen_depth": max(0.08, depth + rng.gauss(0, noise)),
        })
    return items


def fits(outer, inner, observed=False):
    """A simplified container nesting rule, not a full rigid-body simulation."""
    if outer["id"] == inner["id"] or outer["shape"] != inner["shape"]:
        return False
    prefix = "seen_" if observed else "true_"
    return (outer[prefix + "width"] >= inner[prefix + "width"] + CLEARANCE
            and outer[prefix + "depth"] >= inner[prefix + "depth"] + CLEARANCE)


def relations(items, observations=None, observed=True):
    """Pairwise directed feasibility; observations are a probe result cache."""
    observations = observations or {}
    return tuple(tuple(
        False if i == j else observations.get((i, j),
            fits(items[i], items[j], observed=observed))
        for j in range(len(items))) for i in range(len(items)))


def minimum_towers(edges):
    """Exact minimum chain cover for <=7 objects; outer-to-inner paths.

    Graphs from visual estimates are directed partial orders; probe overrides
    may break transitivity, so enumerate simple paths rather than assume it.
    """
    n = len(edges)
    if not 1 <= n <= 7 or any(len(row) != n for row in edges):
        raise ValueError("Expected a square 1-7-item relation graph")
    paths = []
    def walk(path, mask):
        paths.append((tuple(path), mask))
        for candidate in range(n):
            bit = 1 << candidate
            if not mask & bit and edges[path[-1]][candidate]:
                walk(path + [candidate], mask | bit)
    for i in range(n):
        walk([i], 1 << i)
    all_mask = (1 << n) - 1

    @lru_cache(maxsize=None)
    def cover(covered):
        if covered == all_mask:
            return ()
        missing = next(i for i in range(n) if not covered & (1 << i))
        options = []
        for path, mask in paths:
            if mask & (1 << missing) and not mask & covered:
                options.append((path,) + cover(covered | mask))
        return min(options, key=lambda value: (len(value), value))
    return cover(0)


def execute(plan, true_edges):
    """Try proposed nestings; a failed placement starts a separate tower."""
    towers, failures, attempted = [], 0, 0
    for proposed in plan:
        current = [proposed[0]]
        for inside in proposed[1:]:
            attempted += 1
            if true_edges[current[-1]][inside]:
                current.append(inside)
            else:
                failures += 1
                towers.append(tuple(current))
                current = [inside]
        towers.append(tuple(current))
    return {"towers": towers, "failed_joins": failures,
            "attempted_joins": attempted}


def candidate_pairs(items, known):
    return [(i, j) for i in range(len(items)) for j in range(len(items))
            if i != j and items[i]["shape"] == items[j]["shape"]
            and (i, j) not in known]


def proximity(items, pair):
    """Small margin => visually ambiguous proposed fit; no hidden values."""
    i, j = pair
    return min(abs(items[i]["seen_width"] - items[j]["seen_width"] - CLEARANCE),
               abs(items[i]["seen_depth"] - items[j]["seen_depth"] - CLEARANCE))


def select_probe(items, known, method, rng):
    """The active policy tests the most uncertain join in its current plan."""
    candidates = candidate_pairs(items, known)
    if not candidates:
        return None
    if method == "random":
        return rng.choice(candidates)
    if method != "active":
        raise ValueError("Unknown probe policy")
    plan = minimum_towers(relations(items, known))
    planned = [(a, b) for tower in plan for a, b in zip(tower, tower[1:])
               if (a, b) not in known]
    pool = planned if planned else candidates
    return min(pool, key=lambda pair: (proximity(items, pair), pair))


def strategy(items, method, budget, seed):
    if method not in ("vision", "random", "active", "oracle"):
        raise ValueError("Unknown strategy")
    if budget < 0:
        raise ValueError("Probe budget must be nonnegative")
    rng = random.Random(seed)
    known = {}
    true_edges = relations(items, observed=False)
    if method in ("active", "random"):
        for _ in range(budget):
            pair = select_probe(items, known, method, rng)
            if pair is None:
                break
            known[pair] = true_edges[pair[0]][pair[1]]
    edges = true_edges if method == "oracle" else relations(items, known)
    plan = minimum_towers(edges)
    result = execute(plan, true_edges)
    optimum = len(minimum_towers(true_edges))
    return {
        "plan": [list(t) for t in plan],
        "final_towers": [list(t) for t in result["towers"]],
        "probes": [{"outer": i, "inner": j, "fits": value}
                   for (i, j), value in known.items()],
        "probe_count": len(known), "attempted_joins": result["attempted_joins"],
        "failed_joins": result["failed_joins"],
        "tower_count": len(result["towers"]), "oracle_towers": optimum,
        "excess_towers": len(result["towers"]) - optimum,
        "optimal_valid": result["failed_joins"] == 0
            and len(result["towers"]) == optimum,
    }


def score(seed=11, episodes=40, count=6, noise=0.11, budget=3):
    """Paired worlds across methods, with identical item observations."""
    if episodes < 1 or budget < 0 or count < 2 or count > 7:
        raise ValueError("Invalid experiment dimensions")
    methods = ("vision", "random", "active", "oracle")
    records = []
    for index in range(episodes):
        world_seed = seed * 100000 + index
        items = make_items(world_seed, count, noise)
        arms = {m: strategy(items, m, budget, world_seed + 997)
                for m in methods}
        records.append({
            "episode_id": world_seed,
            "visible_items": [{"id": x["id"], "shape": x["shape"],
                "width": round(x["seen_width"], 6),
                "depth": round(x["seen_depth"], 6)} for x in items],
            "arms": arms,
        })
    means = {}
    for method in methods:
        results = [row["arms"][method] for row in records]
        means[method] = {
            "mean_excess_towers": sum(r["excess_towers"] for r in results) / episodes,
            "mean_failed_joins": sum(r["failed_joins"] for r in results) / episodes,
            "optimal_valid_rate": sum(r["optimal_valid"] for r in results) / episodes,
            "mean_probes": sum(r["probe_count"] for r in results) / episodes,
        }
    paired = {
        "active_minus_random_excess": sum(
            row["arms"]["active"]["excess_towers"]
            - row["arms"]["random"]["excess_towers"]
            for row in records) / episodes,
        "active_minus_vision_excess": sum(
            row["arms"]["active"]["excess_towers"]
            - row["arms"]["vision"]["excess_towers"]
            for row in records) / episodes,
    }
    return {
        "protocol": "embodied-organization-v1",
        "settings": {"seed": seed, "episodes": episodes, "count": count,
                     "visual_noise": noise, "probe_budget": budget,
                     "clearance": CLEARANCE},
        "summary": means, "paired_differences": paired, "episodes": records,
        "limitations": "Symbolic scene and exact combinatorial planner. Shapes "
          "and noisy dimensions are supplied, not perceived from pixels. "
          "Touch returns a pairwise boolean, not force measurements. "
          "No learned rules, policy optimization, visual reconstruction, "
          "dynamics model, causal generalization, or physical robotics is claimed.",
    }


def run(out="runs/embodied-organization/report.json", **kwargs):
    path = Path(out)
    if path.exists():
        raise FileExistsError("Use a fresh output path; existing results are preserved")
    report = score(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2) + "\n"
    path.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return {"path": str(path), "sha256": digest,
            "settings": report["settings"], "summary": report["summary"],
            "paired_differences": report["paired_differences"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/embodied-organization/report.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--noise", type=float, default=0.11)
    parser.add_argument("--budget", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.out, seed=args.seed, episodes=args.episodes,
        count=args.count, noise=args.noise, budget=args.budget), indent=2))


if __name__ == "__main__":
    main()
