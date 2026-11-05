"""Two-axis cluster-aware paired effect summaries for synthetic video studies.

Rows are independently initialized models; columns are independent world
seed *pairs*. Opposite histories and correlated pixels are NEVER counted as
independent observations. All model rows evaluate the same scene-pair columns.
"""
import numpy as np


def _inputs(candidate, baseline):
    a = np.asarray(candidate, float)
    b = np.asarray(baseline, float)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("Need >=2 model seeds x >=2 independent scene pairs")
    if b.ndim == 1:
        if b.shape[0] != a.shape[1]:
            raise ValueError("Baseline world-pair IDs must align")
        b = np.broadcast_to(b[None], a.shape)
    if b.shape != a.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Missing/misaligned/nonfinite scores; never silently drop cases")
    return a, b


def paired_effect(candidate, baseline, seed=7, draws=2000,
                  practical_margin=0.10):
    """Cluster bootstrap across model initializations and shared scene pairs.

    Contrast is candidate minus baseline; positive means improvement for
    higher-is-better metrics (e.g. success probability). Resamples entire
    model-seed rows and scene-pair columns, preserving within-pair covariance.
    """
    a, b = _inputs(candidate, baseline)
    if draws < 100 or not 0 <= practical_margin <= 1:
        raise ValueError("Need >=100 draws and valid practical improvement margin")
    delta = a - b
    rng = np.random.default_rng(seed)
    seed_ids = rng.integers(0, delta.shape[0], size=(draws,delta.shape[0]))
    pair_ids = rng.integers(0, delta.shape[1], size=(draws,delta.shape[1]))
    estimates = np.empty(draws)
    for i in range(draws):
        estimates[i] = delta[np.ix_(seed_ids[i], pair_ids[i])].mean()
    low, high = np.quantile(estimates, [.025,.975])
    # Uncertainty interval is not a proof of the null. These labels are
    # conditional on the finite synthetic simulator and frozen protocol.
    if low > practical_margin:
        decision = "supports_practical_advantage"
    elif high < practical_margin:
        decision = "no_supported_practical_advantage"
    else:
        decision = "inconclusive"
    return {
        "absolute_effect": float(delta.mean()),
        "mean_candidate": float(a.mean()),
        "mean_baseline": float(b.mean()),
        "ci_95": [float(low),float(high)],
        "seed_mean_effects": delta.mean(axis=1).tolist(),
        "n_model_seeds": int(a.shape[0]),
        "n_independent_scene_pairs": int(a.shape[1]),
        "bootstrap_draws": draws,
        "practical_margin": practical_margin,
        "decision": decision,
        "method": "two-axis cluster bootstrap; shared scene-pair columns; "
                  "one pair is one unit across two opposing histories",
        "scope": "conditional on this synthetic generator and model-seed population"
    }


def pilot_precision(candidate, baseline, target_model_seeds=(3,5,10),
                    target_scene_pairs=(12,30,60), draws=300, seed=17):
    """Exploratory resampling widths, NOT prospective power analysis.

    Resample pilot empirical effects at proposed sample sizes. This
    estimates plausible interval widths assuming the pilot distribution
    is representative; it cannot show external generalization or promise
    a desired power at an unobserved effect size.
    """
    a,b = _inputs(candidate,baseline)
    if draws < 100:
        raise ValueError("Need >=100 draws")
    delta = a - b
    rng = np.random.default_rng(seed)
    results = []
    for nseed in target_model_seeds:
        for npair in target_scene_pairs:
            if nseed < 2 or npair < 2:
                raise ValueError("Targets require >=2 seeds and pairs")
            estimates = []
            for _ in range(draws):
                rows = rng.integers(0, delta.shape[0], nseed)
                cols = rng.integers(0, delta.shape[1], npair)
                estimates.append(float(delta[np.ix_(rows,cols)].mean()))
            lo,hi = np.quantile(estimates,[.025,.975])
            results.append({"model_seeds":int(nseed),"scene_pairs":int(npair),
                            "pilot_resampled_ci_width":float(hi-lo)})
    return {"scenarios":results,
            "warning":"Resampling precision projections, not power calculations; "
                      "freeze margin and sample plan prior to final test."}
