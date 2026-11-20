"""Phase-conditioned recurrent feedback: controlled synthetic routing probe.

Run: python -m experiments.phase_feedback_eligibility --seeds 11 23 41
Uses NumPy only. Phase is an observed synthetic variable, not inferred brain activity.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def gate(phase, mode, rng):
    """Feedback eligibility; same scalar feedback pathway in every arm."""
    if mode == "phase":
        return 0.5 + 0.5 * np.cos(phase)
    if mode == "shuffled":
        return 0.5 + 0.5 * np.cos(rng.permutation(phase))
    if mode == "constant":
        return np.full_like(phase, 0.5)
    if mode == "learned":
        # Non-phase learned gate control: oracle-tuned constant on validation.
        raise ValueError("learned constant is selected by validation")
    raise ValueError(mode)


def generate(seed, n, phase_informative=True):
    """One-step recurrent feedback toy: a cue is reliable near phase zero.

    Target y is latent sign; feedforward sensory stream is noisy; top-down
    feedback carries a noisy cue whose reliability varies with receiving phase.
    Splits use independent RNG seeds. No future labels enter gate calculation.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, n)
    target = rng.choice([-1.0, 1.0], size=n)
    sensory = target + rng.normal(0, 1.2, n)
    reliability = 0.5 + 0.5 * np.cos(phase) if phase_informative else np.full(n, 0.5)
    feedback = target + rng.normal(0, 0.25 + 2.0 * (1 - reliability), n)
    return phase, target, sensory, feedback


def score(data, eligibility):
    phase, target, sensory, feedback = data
    # One recurrent update with a shared feedback weight, no trainable advantage.
    state = 0.65 * sensory + eligibility * feedback
    return float(np.mean(np.sign(state) != target))


def evaluate(seed, n=6000, phase_informative=True):
    validation = generate(seed + 10000, n, phase_informative)
    test = generate(seed + 20000, n, phase_informative)
    vrng = np.random.default_rng(seed + 30000)
    trng = np.random.default_rng(seed + 40000)
    # Strong control: tune the phase-blind constant gain on validation only.
    candidates = np.linspace(0, 1, 41)
    best = min(candidates, key=lambda c: score(validation, np.full(n, c)))
    result = {
        "phase": score(test, gate(test[0], "phase", trng)),
        "shuffled": score(test, gate(test[0], "shuffled", trng)),
        "constant": score(test, gate(test[0], "constant", trng)),
        "validation_tuned_constant": score(test, np.full(n, best)),
        "sensory_only": score(test, np.zeros(n)),
        "chosen_constant_gain": float(best),
    }
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 41])
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--out", type=Path, default=Path("runs/phase-feedback-eligibility.json"))
    args = p.parse_args(argv)
    if args.samples < 100:
        p.error("--samples must be >= 100")
    rows = []
    for informative in [True, False]:
        for seed in args.seeds:
            rows.append({"seed": seed, "phase_informative": informative,
                         **evaluate(seed, args.samples, informative)})
    report = {"protocol": "synthetic_phase_feedback_v1", "rows": rows,
              "note": "Fixed synthetic phase gate; no learned phase oscillator or neural evidence."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
