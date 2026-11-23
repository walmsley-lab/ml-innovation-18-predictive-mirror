"""Run with python experiments/paired_reveal.py. Requires NumPy only."""
import json
import numpy as np


def observe(domain, bit, reveal):
    if domain == "visual":
        # Central pixel is a light or dark mark. A screen covers the entire patch.
        patch = np.zeros((3, 3))
        if reveal:
            patch[:] = 0.5
            patch[1, 1] = bit
        return patch.ravel()
    # Only inspection exposes the queried directed edge and its observation mask.
    return np.array([reveal, bit if reveal else 0], dtype=float)


def episodes(domain, delays):
    for delay in delays:
        for inspect in (False, True):
            for bit in (0, 1):
                phases = ["before", "inspect" if inspect else "sham"] + ["hidden"] * delay
                views = [observe(domain, bit, phase == "inspect") for phase in phases]
                yield bit, phases, views


def samples(domain, delays, remember):
    for bit, phases, views in episodes(domain, delays):
        memory = np.zeros_like(views[0])
        for i, (phase, view) in enumerate(zip(phases, views)):
            if phase == "inspect":
                memory = view.copy()
            features = np.r_[1, view, memory] if remember else np.r_[1, view]
            if i == len(phases) - 1:
                phase = "retained" if phases[1] == "inspect" else "never_revealed"
            yield features, bit, phase


def evaluate(domain, remember):
    train = list(samples(domain, range(1, 5), remember))
    weights = np.linalg.lstsq(np.stack([s[0] for s in train]),
                              [s[1] for s in train], rcond=None)[0]
    errors = {}
    for features, bit, phase in samples(domain, (6, 9, 12), remember):
        probability = np.clip(features @ weights, 0, 1)
        errors.setdefault(phase, []).append((probability - bit) ** 2)
    return {phase: round(float(np.mean(values)), 6) for phase, values in errors.items()
            if phase != "hidden"}


if __name__ == "__main__":
    results = {}
    for domain in ("visual", "graph"):
        for reveal in (False, True):
            differ = np.any(observe(domain, 0, reveal) != observe(domain, 1, reveal))
            assert differ == reveal
        for remember in (False, True):
            scores = evaluate(domain, remember)
            expected = {"before": 0.25, "sham": 0.25, "inspect": 0.0,
                        "never_revealed": 0.25, "retained": 0.0 if remember else 0.25}
            assert scores == expected, (domain, remember, scores)
            results[f"{domain}-{'memory' if remember else 'current'}"] = scores
    print(json.dumps(results, indent=2))
