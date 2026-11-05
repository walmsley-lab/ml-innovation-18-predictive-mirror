"""Roadmap 31: split-conformal visual-position forecast intervals.

A small scalar observed-position proxy isolates calibration from rendering.
All forecast inputs come from an observed history; targets are withheld until
calibration or evaluation. Split by complete independent episodes.
"""
import argparse
import json
from pathlib import Path
import numpy as np

ALPHA = .10
HORIZON = 5


def episode(seed, context=7, horizon=HORIZON):
    """Observed noisy positions and future observed positions (label only)."""
    if context < 5 or horizon < 1:
        raise ValueError("Need observable motion and a forecast")
    rng = np.random.default_rng(int(seed))
    scale = float(rng.choice([.12, .5, 1.2]))
    velocity = float(rng.uniform(-1.5, 1.5))
    offset = float(rng.uniform(-4, 4))
    past = offset + velocity * np.arange(context)
    past += rng.normal(0, scale, context)
    leads = np.arange(1, horizon + 1, dtype=float)
    future = offset + velocity * (context - 1 + leads)
    future += rng.normal(0, scale * np.sqrt(leads), horizon)
    # Scale, true velocity and noise are never returned to the estimator.
    return past, future


def forecast(past, horizon=HORIZON):
    past = np.asarray(past, float)
    if past.ndim != 1 or len(past) < 5 or horizon < 1:
        raise ValueError("Need one observed history and positive horizon")
    differences = np.diff(past)
    speed = float(np.median(differences))
    volatility = float(np.median(np.abs(differences - speed)))
    leads = np.arange(1, horizon + 1)
    return past[-1] + speed * leads, volatility


def examples(start_seed, count, horizon=HORIZON):
    if count < 1:
        raise ValueError("Need independent episodes")
    predictions, actual, volatility = [], [], []
    for seed in range(start_seed, start_seed + count):
        past, truth = episode(seed, horizon=horizon)
        pred, vol = forecast(past, horizon)
        predictions.append(pred)
        actual.append(truth)
        volatility.append(vol)
    return np.asarray(predictions), np.asarray(actual), np.asarray(volatility)


def quantile(scores, alpha=ALPHA):
    """Conformal rank, including n+1 finite-sample correction."""
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not len(scores) or not 0 < alpha < 1:
        raise ValueError("Need scores and 0<alpha<1")
    rank = int(np.ceil((len(scores) + 1) * (1-alpha)))
    if rank > len(scores):
        raise ValueError("Too few calibration episodes for this alpha")
    return float(np.sort(scores)[rank-1])


def calibrate(predicted, truth, volatility, boundaries, alpha=ALPHA):
    """Independent calibration episodes; group thresholds fixed from training."""
    predicted, truth, volatility = (np.asarray(x) for x in
                                   (predicted, truth, volatility))
    if predicted.ndim != 2 or predicted.shape != truth.shape or (
            len(volatility) != len(predicted)):
        raise ValueError("Calibration shape mismatch")
    labels = np.digitize(volatility, boundaries)
    errors = np.abs(predicted - truth)
    global_radius = np.array([quantile(errors[:, lead], alpha)
                              for lead in range(errors.shape[1])])
    grouped = np.empty((3, errors.shape[1]), float)
    counts = []
    for group in range(3):
        mask = labels == group
        count = int(mask.sum())
        counts.append(count)
        # A sparse bin borrows the global quantile instead of claiming
        # conditional coverage from too few independent episodes.
        grouped[group] = ([quantile(errors[mask, h], alpha)
                          for h in range(errors.shape[1])]
                          if count >= max(10, int(np.ceil(1 / alpha))) else
                          global_radius)
    return {"global": global_radius, "grouped": grouped, "counts": counts}


def evaluate(predicted, truth, volatility, boundaries, radii, alpha=ALPHA):
    labels = np.digitize(volatility, boundaries)
    delta = np.abs(np.asarray(predicted) - np.asarray(truth))
    result = {}
    for name, width in (("global", np.broadcast_to(radii["global"], delta.shape)),
                        ("conditional", radii["grouped"][labels]),
                        ("fixed_half_pixel", np.full(delta.shape, .5))):
        violation = np.maximum(delta - width, 0.)
        scores = 2*width + (2 / alpha)*violation
        covered = delta <= width
        result[name] = {
            "coverage": float(np.mean(covered)),
            "coverage_per_lead": np.mean(covered, axis=0).tolist(),
            "mean_interval_width": float(np.mean(2 * width)),
            "mean_interval_score": float(np.mean(scores)),
            "coverage_by_volatility_bin": [
                float(np.mean(covered[labels==i])) if np.any(labels==i) else None
                for i in range(3)]}
    return result


def run(out="runs/roadmap-31/metrics.json", training=120, calibration=150,
        testing=120, horizon=HORIZON):
    if min(training, calibration, testing) < 30 or horizon < 1:
        raise ValueError("Need independent data and sufficient calibration")
    _, _, train_vol = examples(31000, training, horizon)
    boundaries = np.quantile(train_vol, [1/3, 2/3])
    cp, ct, cv = examples(61000, calibration, horizon)
    radii = calibrate(cp, ct, cv, boundaries)
    tp, tt, tv = examples(91000, testing, horizon)
    result = {
        "roadmap_id": 31, "unit": "observed scalar position, not RGB pixel MSE",
        "alpha": ALPHA, "horizon": horizon,
        "training_episodes": training, "calibration_episodes": calibration,
        "test_episodes": testing, "training_selected_boundaries": boundaries.tolist(),
        "calibration_bin_counts": radii["counts"],
        "calibration_radii": {k: v.tolist() for k, v in radii.items() if k != "counts"},
        "test": evaluate(tp, tt, tv, boundaries, radii),
        "limitations": "Split-conformal marginal coverage requires exchangeable "
                       "episodes. Conditional bins are exploratory; no guarantee "
                       "under domain shift. This is a scalar position proxy, not "
                       "learned RGB uncertainty or a multimodal video model."}
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/roadmap-31/metrics.json")
    parser.add_argument("--training", type=int, default=120)
    parser.add_argument("--calibration", type=int, default=150)
    parser.add_argument("--testing", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.training, args.calibration,
                         args.testing)["test"], indent=2))


if __name__ == "__main__":
    main()
